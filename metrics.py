"""Scoring and aggregation utilities for the ACID benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

NOISE_TIERS = (8_000, 32_000, 128_000, 512_000)
DOMAIN_TAGS = ("ACID-Math", "ACID-Chem", "ACID-Bio", "ACID-Code")


def _normalise(value: str, *, case_sensitive: bool = False) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value if case_sensitive else value.casefold()


def extract_final_answer(response: str) -> str:
    """Extract a model's marked final answer, falling back to the full response."""
    matches = re.findall(r"<final>\s*(.*?)\s*</final>", response, flags=re.I | re.S)
    return matches[-1] if matches else response.strip()


def score_response(response: str, evaluation: Mapping[str, Any]) -> float:
    """Score a response in [0, 1] with a deterministic evaluation specification.

    Supported evaluator types are exact, contains, regex, numeric, and any_of.
    The model is asked to wrap its answer in <final>, but unwrapped output is valid.
    """
    kind = str(evaluation.get("type", "exact"))
    candidate = extract_final_answer(response)
    case_sensitive = bool(evaluation.get("case_sensitive", False))

    if kind == "any_of":
        raw_alternatives = evaluation.get("alternatives", [])
        if not isinstance(raw_alternatives, list) or not raw_alternatives:
            raise ValueError("any_of evaluation requires a non-empty alternatives list")
        alternatives = cast(list[Mapping[str, Any]], raw_alternatives)
        return max(
            score_response(response, alternative) for alternative in alternatives
        )

    expected = evaluation.get("answer")
    if expected is None:
        raise ValueError(f"{kind} evaluation requires an answer")

    if kind == "exact":
        return float(
            _normalise(candidate, case_sensitive=case_sensitive)
            == _normalise(str(expected), case_sensitive=case_sensitive)
        )

    if kind == "contains":
        haystack = candidate if case_sensitive else candidate.casefold()
        answers = cast(list[Any], expected) if isinstance(expected, list) else [expected]
        needles = [
            str(value) if case_sensitive else str(value).casefold() for value in answers
        ]
        mode = evaluation.get("match", "all")
        checks = [needle in haystack for needle in needles]
        return float(any(checks) if mode == "any" else all(checks))

    if kind == "regex":
        flags = 0 if case_sensitive else re.I
        return float(
            re.search(str(expected), candidate, flags=flags | re.S) is not None
        )

    if kind == "numeric":
        numbers = re.findall(
            r"[-+]?(?:\d+(?:,\d{3})*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", candidate
        )
        if not numbers:
            return 0.0
        actual = float(numbers[-1].replace(",", ""))
        target = float(expected)
        absolute = float(evaluation.get("absolute_tolerance", 0.0))
        relative = float(evaluation.get("relative_tolerance", 0.0))
        return float(math.isclose(actual, target, rel_tol=relative, abs_tol=absolute))

    raise ValueError(f"Unsupported evaluation type: {kind}")


def _mean(values: Iterable[float]) -> float | None:
    materialised = list(values)
    return statistics.fmean(materialised) if materialised else None


def _tier_label(tier: int | None) -> str | None:
    if tier is None:
        return None
    return "0" if tier == 0 else f"{tier // 1000}K"


def _compliance(
    tier_rows: Iterable[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    passing = [
        int(row["tier_tokens"])
        for row in tier_rows
        if row.get("conservation_ratio") is not None
        and float(row["conservation_ratio"]) >= threshold
    ]
    maximum = max(passing, default=None)
    return {
        "tier_tokens": maximum,
        "tier_label": _tier_label(maximum),
        "threshold": threshold,
    }


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach per-trial ratios and calculate model/domain/tier summaries.

    A noisy score is divided by the baseline for the same model, item, and trial.
    Ratios are null when that baseline is zero; those observations are excluded from
    ratio means while their scores remain present in mean-score calculations.
    """
    baselines: dict[tuple[str, str, str, int], float] = {}
    for row in records:
        if int(row["tier_tokens"]) == 0:
            key = (
                str(row.get("provider", "unknown")),
                str(row["model"]),
                str(row["item_id"]),
                int(row["trial"]),
            )
            baselines[key] = float(row["score"])

    for row in records:
        key = (
            str(row.get("provider", "unknown")),
            str(row["model"]),
            str(row["item_id"]),
            int(row["trial"]),
        )
        baseline = baselines.get(key)
        row["baseline_score"] = baseline
        if int(row["tier_tokens"]) == 0:
            row["conservation_ratio"] = 1.0 if baseline and baseline > 0 else None
        else:
            row["conservation_ratio"] = (
                float(row["score"]) / baseline
                if baseline is not None and baseline > 0
                else None
            )

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        tier = int(row["tier_tokens"])
        provider = str(row.get("provider", "unknown"))
        grouped[(provider, str(row["model"]), str(row["domain"]), tier)].append(row)
        overall[(provider, str(row["model"]), tier)].append(row)

    def summarise(
        key: tuple[Any, ...], rows: list[dict[str, Any]], domain: str
    ) -> dict[str, Any]:
        ratios = [
            float(row["conservation_ratio"])
            for row in rows
            if row["conservation_ratio"] is not None
        ]
        scores = [float(row["score"]) for row in rows]
        return {
            "provider": str(key[0]),
            "model": str(key[1]),
            "domain": domain,
            "tier_tokens": int(key[-1]),
            "tier_label": _tier_label(int(key[-1])),
            "mean_score": _mean(scores),
            "conservation_ratio": _mean(ratios),
            "observations": len(rows),
            "valid_ratios": len(ratios),
            "errors": sum(row.get("status") != "ok" for row in rows),
        }

    domain_rows = [summarise(key, rows, key[2]) for key, rows in sorted(grouped.items())]
    overall_rows = [
        summarise(key, rows, "ALL") for key, rows in sorted(overall.items())
    ]

    compliance: list[dict[str, Any]] = []
    models = sorted({(str(row.get("provider", "unknown")), str(row["model"])) for row in records})
    for provider, model in models:
        model_rows = [
            row
            for row in overall_rows
            if row["provider"] == provider
            and row["model"] == model
            and int(row["tier_tokens"]) > 0
        ]
        compliance.append(
            {
                "provider": provider,
                "model": model,
                "domain": "ALL",
                "ACID80": _compliance(model_rows, 0.80),
                "ACID90": _compliance(model_rows, 0.90),
            }
        )
        for domain in DOMAIN_TAGS:
            rows = [
                row
                for row in domain_rows
                if row["provider"] == provider
                and row["model"] == model
                and row["domain"] == domain
                and int(row["tier_tokens"]) > 0
            ]
            if rows:
                compliance.append(
                    {
                        "provider": provider,
                        "model": model,
                        "domain": domain,
                        "ACID80": _compliance(rows, 0.80),
                        "ACID90": _compliance(rows, 0.90),
                    }
                )

    return {
        "definition": "mean of per-trial (noise score / matching clean baseline score)",
        "zero_baseline_policy": "ratio is null and excluded from ratio means",
        "tier_summary": overall_rows,
        "domain_summary": domain_rows,
        "compliance": compliance,
    }


def write_summary_csv(metrics: Mapping[str, Any], path: Path) -> None:
    """Write overall and domain summaries in one graph-ready long-form CSV."""
    compliance_lookup = {
        (row["provider"], row["model"], row["domain"]): row
        for row in metrics.get("compliance", [])
    }
    rows = list(metrics.get("tier_summary", [])) + list(
        metrics.get("domain_summary", [])
    )
    fieldnames = [
        "provider",
        "model",
        "domain",
        "tier_tokens",
        "tier_label",
        "mean_score",
        "conservation_ratio",
        "observations",
        "valid_ratios",
        "errors",
        "acid80_tokens",
        "acid80_label",
        "acid90_tokens",
        "acid90_label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            compliance = compliance_lookup.get(
                (row["provider"], row["model"], row["domain"]), {}
            )
            writer.writerow(
                {
                    **row,
                    "acid80_tokens": compliance.get("ACID80", {}).get("tier_tokens"),
                    "acid80_label": compliance.get("ACID80", {}).get("tier_label"),
                    "acid90_tokens": compliance.get("ACID90", {}).get("tier_tokens"),
                    "acid90_label": compliance.get("ACID90", {}).get("tier_label"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute ACID metrics from a result JSON file"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output-json", type=Path, default=Path("results/metrics.json")
    )
    parser.add_argument("--output-csv", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = cast(list[dict[str, Any]], payload["records"] if isinstance(payload, dict) else payload)
    calculated = calculate_metrics(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(calculated, indent=2), encoding="utf-8")
    write_summary_csv(calculated, args.output_csv)


if __name__ == "__main__":
    main()
