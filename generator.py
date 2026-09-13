"""Build deterministic, token-exact static noise prefixes for ACID."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import tiktoken

DEFAULT_TIERS = (8_000, 32_000, 128_000, 512_000)
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ACID-Benchmark/1.0 (research evaluation corpus generator)"
NEUTRAL_PADDING = " Historical context and descriptive background are provided for reference."


@dataclass(frozen=True)
class Article:
    page_id: str
    title: str
    text: str
    source_url: str


def clean_article_tail(text: str) -> str:
    """Remove markup and interrogative syntax from an article's tail.

    The whole article is normalised, then trailing question-like lines/sentences are
    removed. This is repeated so a sequence of questions cannot remain at the tail.
    """
    text = html.unescape(text).replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = re.sub(r"(?:^|\n)\s*(?:question|questions|q)\s*:\s*[^\n]*\??\s*$", "", text, flags=re.I)
        text = re.sub(r"(?:^|(?<=[.!]))\s*[^.!\n]*\?\s*$", "", text, flags=re.S)
        text = re.sub(r"[?¿]+\s*$", ".", text).rstrip()
    return text


def _fetch_random_batch(batch_size: int, timeout: float) -> list[Article]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(min(batch_size, 20)),
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
    }
    request = urllib.request.Request(
        f"{WIKIPEDIA_API}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    articles: list[Article] = []
    for page in payload.get("query", {}).get("pages", []):
        cleaned = clean_article_tail(page.get("extract", ""))
        if len(cleaned) >= 500:
            articles.append(
                Article(
                    page_id=str(page["pageid"]),
                    title=page["title"],
                    text=cleaned,
                    source_url=page.get("fullurl", ""),
                )
            )
    return articles


def wikipedia_articles(seed: int, timeout: float = 30.0) -> Iterator[Article]:
    """Yield unique Wikipedia articles in deterministic local shuffle batches.

    Wikipedia's random selection itself is not reproducible. Reproducibility is
    achieved by persisting downloaded sources; the seed fixes their local order.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    while True:
        batch = _fetch_random_batch(20, timeout)
        rng.shuffle(batch)
        for article in batch:
            if article.page_id not in seen:
                seen.add(article.page_id)
                yield article


def local_articles(source_dir: Path, seed: int) -> Iterator[Article]:
    """Yield shuffled, disjoint UTF-8 text files from a local corpus."""
    paths = sorted(source_dir.rglob("*.txt"))
    if not paths:
        raise ValueError(f"No .txt files found under {source_dir}")
    random.Random(seed).shuffle(paths)
    for path in paths:
        cleaned = clean_article_tail(path.read_text(encoding="utf-8"))
        if cleaned:
            yield Article(
                page_id=str(path.resolve()),
                title=path.stem,
                text=cleaned,
                source_url=path.resolve().as_uri(),
            )


def _load_cached_sources(path: Path) -> list[Article]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Article(**row) for row in payload]


def _safe_exact_prefix(tokens: list[int], budget: int, encoding: tiktoken.Encoding) -> str:
    """Return exactly budget tokens and guarantee a non-interrogative tail."""
    selected = tokens[:budget]
    tail = encoding.decode(selected[-256:])
    unsafe = bool(
        re.search(r"(?:\?|¿)\s*$", tail)
        or re.search(r"(?:^|\n)\s*(?:question|questions|q)\s*:", tail, flags=re.I)
    )
    if unsafe:
        padding = encoding.encode(NEUTRAL_PADDING * 64)
        replace_count = min(256, budget)
        selected[-replace_count:] = padding[:replace_count]
    text = encoding.decode(selected)
    # tiktoken token sequences round-trip; fail loudly if a custom encoding does not.
    if len(encoding.encode(text)) != budget:
        raise RuntimeError("Unable to produce an exact token budget with this encoding")
    if re.search(r"[?¿]\s*$", text):
        raise RuntimeError("Noise sanitisation left interrogation syntax at the block tail")
    return text


def compile_noise_blocks(
    articles: Iterable[Article],
    output_dir: Path,
    tiers: tuple[int, ...] = DEFAULT_TIERS,
    encoding_name: str = "cl100k_base",
    source_cache: Path | None = None,
) -> dict[str, object]:
    """Compile nested static prefixes, making smaller tiers prefixes of larger ones."""
    if not tiers or any(tier <= 0 for tier in tiers):
        raise ValueError("Tiers must be positive token counts")
    tiers = tuple(sorted(set(tiers)))
    encoding = tiktoken.get_encoding(encoding_name)
    required = max(tiers) + 512
    token_stream: list[int] = []
    used: list[Article] = []

    for article in articles:
        separator = f"\n\n--- Reference article: {article.title} ---\n"
        token_stream.extend(encoding.encode(separator + article.text))
        used.append(article)
        if len(token_stream) >= required:
            break
    if len(token_stream) < required:
        raise RuntimeError(
            f"Corpus exhausted at {len(token_stream):,} tokens; need at least {required:,}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for tier in tiers:
        noise = _safe_exact_prefix(token_stream, tier, encoding)
        # Propagate any safe-tail replacement into larger tiers so every smaller
        # file remains a byte-for-byte token prefix of every larger file.
        token_stream[:tier] = encoding.encode(noise)
        path = output_dir / f"noise_{tier}.txt"
        path.write_text(noise, encoding="utf-8", newline="")
        files[str(tier)] = {
            "path": path.name,
            "tokens": len(encoding.encode(noise)),
            "characters": len(noise),
            "sha256": hashlib.sha256(noise.encode("utf-8")).hexdigest(),
        }

    cache_path = source_cache or output_dir / "sources.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(article) for article in used], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "encoding": encoding_name,
        "nested_prefixes": True,
        "article_count": len(used),
        "source_cache": str(cache_path),
        "tiers": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_tiers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.replace("_", "")) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tiers must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token-exact static ACID noise files")
    parser.add_argument("--output-dir", type=Path, default=Path("noise_cache"))
    parser.add_argument("--source-dir", type=Path, help="Use local .txt corpus instead of Wikipedia")
    parser.add_argument("--source-cache", type=Path, help="Reuse or write downloaded source articles")
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--tiers", type=_parse_tiers, default=DEFAULT_TIERS)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    cached = _load_cached_sources(args.source_cache) if args.source_cache else []
    if cached:
        sources: Iterable[Article] = cached
    elif args.source_dir:
        sources = local_articles(args.source_dir, args.seed)
    else:
        sources = wikipedia_articles(args.seed)

    manifest = compile_noise_blocks(
        sources,
        args.output_dir,
        tiers=args.tiers,
        encoding_name=args.encoding,
        source_cache=args.source_cache,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
