"""Asynchronous ACID runner with separate benchmark and scoring-model paths."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from metrics import DOMAIN_TAGS, calculate_metrics, write_summary_csv

DEFAULT_TIERS = (8_000, 32_000, 128_000, 512_000)
DEFAULT_JUDGE_MODEL_ID = "gpt-4o-mini"
JUDGE_INSTRUCTIONS_VERSION = "acid-judge-v1"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256

ANCHOR = """

========== ACID TARGET TASK ==========
Ignore the reference material above. Solve only the target task below. Show any
reasoning that is useful, then place only the requested final answer between
<final> and </final> tags.

{question}
========== END TARGET TASK ==========
"""

JUDGE_SYSTEM_PROMPT = """You are the standardized ACID scoring model. Follow this rubric exactly.
1. Judge only whether the candidate answers the supplied question correctly.
2. Compare against the supplied reference answer; do not invent missing facts.
3. Treat equivalent wording, numeric forms, units, and valid code outputs as correct.
4. Ignore verbosity, formatting, and style unless they make the answer incorrect.
5. Use 1.0 for fully correct, 0.0 for fully incorrect, and a value between them only
    when the response contains meaningful partial correctness.
6. Return only JSON in this exact shape: {"score": 0.0, "rationale": "brief explanation"}.
The score must be a number from 0.0 to 1.0."""

JUDGE_PROMPT = """Evaluate this candidate response using the standardized ACID rubric.

COMPLETE QUESTION ITEM (there is deliberately no injected noise here):
{item}

CANDIDATE RESPONSE FROM THE BENCHMARKED MODEL:
{response}
"""


@dataclass(frozen=True)
class ModelSpec:
    """An OpenAI-compatible model entry from models.json or the CLI."""

    id: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    http_referer: str | None = None
    app_title: str | None = None


@dataclass(frozen=True)
class RunConfig:
    trials: int
    concurrency: int
    max_tokens: int
    temperature: float
    timeout: float
    retries: int
    dry_run: bool = False


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _read_question_file(path: Path) -> list[dict[str, Any]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    raw_items: list[Any]
    if isinstance(payload, list):
        raw_items = cast(list[Any], payload)
    elif isinstance(payload, dict) and isinstance(cast(dict[str, Any], payload).get("items"), list):
        raw_items = cast(list[Any], cast(dict[str, Any], payload)["items"])
    else:
        raw_items = [payload]
    if not all(isinstance(item, dict) for item in raw_items):
        raise ValueError(f"Every item in {path} must be an object")
    return [dict(cast(dict[str, Any], item)) for item in raw_items]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load one question JSON file or every JSON file in a questions directory."""
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise ValueError(f"No .json question files found under {path}")
        items = [item for file in files for item in _read_question_file(file)]
    else:
        items = _read_question_file(path)
    validate_dataset(items)
    return items


def validate_dataset(items: Any) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("Question set must contain at least one item")
    raw_items = cast(list[Any], items)
    if not all(isinstance(item, dict) for item in raw_items):
        raise ValueError("Every question item must be an object")
    typed_items = cast(list[dict[str, Any]], raw_items)
    seen: set[str] = set()
    for index, item in enumerate(typed_items):
        missing = {"id", "domain", "question", "answer"} - item.keys()
        if missing:
            raise ValueError(f"Item {index} is missing: {', '.join(sorted(missing))}")
        unexpected = set(item) - {"id", "domain", "question", "answer", "difficulty"}
        if unexpected:
            raise ValueError(f"Item {index} has unsupported fields: {', '.join(sorted(unexpected))}")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate item id: {item_id}")
        seen.add(item_id)
        if item["domain"] not in DOMAIN_TAGS:
            raise ValueError(f"Unknown domain on {item_id}: {item['domain']}")
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Empty question on {item_id}")
        # Reserved for future benchmark analysis; intentionally unused for now.
        if "difficulty" in item and not isinstance(item["difficulty"], str):
            raise ValueError(f"Difficulty must be a string on {item_id}")


def _spec_from_json(raw: str | Mapping[str, Any]) -> ModelSpec:
    return ModelSpec(id=raw) if isinstance(raw, str) else ModelSpec(**raw)


def _load_registry(path: Path) -> tuple[dict[str, ModelSpec], ModelSpec | None]:
    if not path.exists():
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = {_spec_from_json(raw).id: _spec_from_json(raw) for raw in payload.get("models", [])}
    scoring = payload.get("scoring_model")
    return models, (_spec_from_json(scoring) if scoring is not None else None)


def load_model_specs(path: Path, requested_models: Sequence[str] | None) -> list[ModelSpec]:
    """Resolve benchmark model IDs; add another model by editing models.json or the CLI."""
    configured, _ = _load_registry(path)
    ids = list(requested_models or configured)
    if not ids:
        raise ValueError("Add benchmark models to models.json or pass --models MODEL [MODEL ...]")
    return [configured.get(model_id, ModelSpec(id=model_id)) for model_id in ids]


def load_scoring_spec(path: Path, requested_model: str | None, *, dry_run: bool) -> ModelSpec:
    configured, registry_scoring = _load_registry(path)
    if requested_model:
        return configured.get(requested_model, ModelSpec(id=requested_model))
    if registry_scoring is not None:
        return registry_scoring
    return ModelSpec(id=DEFAULT_JUDGE_MODEL_ID)


def load_noise(noise_dir: Path, requested_tiers: Sequence[int] | None) -> tuple[dict[int, str], str]:
    manifest_path = noise_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoding_name = str(manifest["encoding"])
    encoding = tiktoken.get_encoding(encoding_name)
    available = {int(tier): details for tier, details in manifest["tiers"].items()}
    tiers = sorted(requested_tiers or available.keys())
    blocks: dict[int, str] = {0: ""}
    for tier in tiers:
        if tier not in available:
            raise ValueError(f"Tier {tier} is absent from {manifest_path}")
        details = available[tier]
        text = (noise_dir / details["path"]).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != details["sha256"]:
            raise ValueError(f"Noise file checksum mismatch for tier {tier}")
        if len(encoding.encode(text)) != tier:
            raise ValueError(f"Noise tier {tier} contains the wrong token count")
        blocks[tier] = text
    return blocks, encoding_name


def dry_noise_blocks(requested_tiers: Sequence[int] | None) -> tuple[dict[int, str], str]:
    tiers = sorted(requested_tiers or DEFAULT_TIERS)
    return {0: "", **{tier: f"[DRY RUN NOISE TIER {tier}]" for tier in tiers}}, "dry-run"


def build_payload(noise: str, question: str) -> str:
    """Keep the static noise first and anchor the question at the end."""
    return noise + ANCHOR.format(question=question.strip())


def _answer_text(answer: Any) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False, sort_keys=True)


def build_judge_payload(item: Mapping[str, Any], response: str) -> str:
    """Build judge input from question/domain/answer only; never include noise."""
    judge_item = {"question": item["question"], "domain": item["domain"], "answer": item["answer"]}
    return JUDGE_PROMPT.format(item=json.dumps(judge_item, ensure_ascii=False, indent=2), response=response)


def parse_judge_response(response: str) -> tuple[float, str | None]:
    cleaned = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.I | re.S)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\"']score[\"']\s*:\s*([01](?:\.\d+)?)", cleaned, flags=re.I)
        if not match:
            raise ValueError("Scoring model did not return JSON containing score")
        return float(match.group(1)), None
    score = float(payload["score"])
    if not 0 <= score <= 1:
        raise ValueError("Scoring model score must be between 0 and 1")
    rationale = payload.get("rationale")
    return score, str(rationale) if rationale is not None else None


class AcidRunner:
    def __init__(self, model_clients: Mapping[str, AsyncOpenAI], models: Sequence[ModelSpec], scoring_client: AsyncOpenAI | None,
                 scoring_model: ModelSpec, items: Sequence[Mapping[str, Any]], noise_blocks: Mapping[int, str], config: RunConfig) -> None:
        self.model_clients = model_clients
        self.models = models
        self.scoring_client = scoring_client
        self.scoring_model = scoring_model
        self.items = items
        self.noise_blocks = noise_blocks
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrency)

    async def _request(self, client: AsyncOpenAI, model: str, content: str, *, system_prompt: str | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> tuple[str, dict[str, Any], int, float]:
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            started = time.perf_counter()
            try:
                messages: list[ChatCompletionMessageParam] = []
                if system_prompt:
                    messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt}))
                messages.append(cast(ChatCompletionMessageParam, {"role": "user", "content": content}))
                response = await client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                    timeout=self.config.timeout,
                )
                usage = response.usage.model_dump() if response.usage else {}
                return response.choices[0].message.content or "", usage, attempt + 1, time.perf_counter() - started
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.retries:
                    break
                await asyncio.sleep(min(30.0, 2**attempt + random.random()))
        assert last_error is not None
        raise last_error

    async def _judge(self, item: Mapping[str, Any], response: str) -> tuple[float, str, dict[str, Any], int, float, str | None]:
        if self.config.dry_run:
            return 1.0, '{"score": 1.0, "rationale": "dry run"}', {}, 0, 0.0, "dry run"
        if self.scoring_client is None:
            raise RuntimeError("A scoring client is required outside dry-run mode")
        raw, usage, attempts, latency = await self._request(
            self.scoring_client,
            self.scoring_model.id,
            build_judge_payload(item, response),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        score, rationale = parse_judge_response(raw)
        return score, raw, usage, attempts, latency, rationale

    async def _run_one(self, model: ModelSpec, item: Mapping[str, Any], tier: int, trial: int) -> dict[str, Any]:
        record: dict[str, Any] = {
            "model": model.id, "scoring_model": self.scoring_model.id, "item_id": item["id"],
            "domain": item["domain"], "tier_tokens": tier, "trial": trial, "status": "error",
            "score": 0.0, "response": None, "judge_response": None, "judge_rationale": None,
            "error": None, "usage": {}, "judge_usage": {}, "attempts": 0, "judge_attempts": 0,
            "latency_seconds": None, "judge_latency_seconds": None,
        }
        async with self.semaphore:
            try:
                if self.config.dry_run:
                    response, usage, attempts, latency = f"<final>{_answer_text(item['answer'])}</final>", {}, 0, 0.0
                else:
                    response, usage, attempts, latency = await self._request(
                        self.model_clients[model.id], model.id, build_payload(self.noise_blocks[tier], item["question"])
                    )
                score, judge_response, judge_usage, judge_attempts, judge_latency, rationale = await self._judge(item, response)
                record.update({
                    "status": "ok", "score": score, "response": response, "judge_response": judge_response,
                    "judge_rationale": rationale, "usage": usage, "judge_usage": judge_usage,
                    "attempts": attempts, "judge_attempts": judge_attempts,
                    "latency_seconds": latency, "judge_latency_seconds": judge_latency,
                })
            except Exception as exc:
                record["attempts"] = self.config.retries + 1
                record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    async def run(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tier in sorted(self.noise_blocks):
            tasks = [self._run_one(model, item, tier, trial) for model in self.models for item in self.items for trial in range(1, self.config.trials + 1)]
            records.extend(await asyncio.gather(*tasks))
        return records


def _parse_int_list(value: str) -> list[int]:
    try:
        return [int(part.replace("_", "")) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _make_client(spec: ModelSpec, env_file: Path) -> AsyncOpenAI:
    load_dotenv(env_file)
    api_key = os.getenv(spec.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {spec.api_key_env} for model {spec.id}")
    headers: dict[str, str] = {}
    if spec.http_referer:
        headers["HTTP-Referer"] = spec.http_referer
    if spec.app_title:
        headers["X-Title"] = spec.app_title
    return AsyncOpenAI(api_key=api_key, base_url=spec.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), default_headers=headers)


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file)
    items = load_dataset(args.dataset)
    models = load_model_specs(args.model_config, args.models)
    scoring_model = load_scoring_spec(args.model_config, args.scoring_model, dry_run=args.dry_run)
    if args.dry_run:
        noise_blocks, encoding_name = (load_noise(args.noise_dir, args.tiers) if (args.noise_dir / "manifest.json").exists() else dry_noise_blocks(args.tiers))
        clients: dict[str, AsyncOpenAI] = {}
        scoring_client = None
    else:
        noise_blocks, encoding_name = load_noise(args.noise_dir, args.tiers)
        clients = {model.id: _make_client(model, args.env_file) for model in models}
        scoring_client = _make_client(scoring_model, args.env_file)
    config = RunConfig(args.trials, args.concurrency, args.max_tokens, args.temperature, args.timeout, args.retries, args.dry_run)
    records = await AcidRunner(clients, models, scoring_client, scoring_model, items, noise_blocks, config).run()
    metrics = calculate_metrics(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": "ACID", "format_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {"models": [asdict(model) for model in models], "scoring_model": asdict(scoring_model),
                  "judge_instructions_version": JUDGE_INSTRUCTIONS_VERSION,
                  "judge_temperature": JUDGE_TEMPERATURE, "judge_max_tokens": JUDGE_MAX_TOKENS,
                  "tiers": sorted(noise_blocks), "trials": args.trials, "encoding": encoding_name, "dry_run": args.dry_run},
        "records": records, "metrics": metrics,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(metrics, args.output_dir / "summary.csv")
    print(f"Wrote {len(records)} observations to {result_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ACID with benchmark models and a separate AI scoring model")
    parser.add_argument("--dataset", type=Path, default=Path("questions"), help="Question JSON file or folder")
    parser.add_argument("--model-config", type=Path, default=Path("models.json"), help="Editable model registry")
    parser.add_argument("--noise-dir", type=Path, default=Path("noise_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="+", help="Benchmark model IDs; omit to use models.json")
    parser.add_argument("--scoring-model", help="Optional non-comparable override; default is the standardized judge in models.json or gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true", help="Run without constructing or calling any AI provider client")
    parser.add_argument("--tiers", type=_parse_int_list, help="Comma-separated subset; default: manifest tiers")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials < 1 or args.concurrency < 1 or args.retries < 0:
        raise SystemExit("trials/concurrency must be positive and retries cannot be negative")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
