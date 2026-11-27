"""Measure indexing time and keyword query latency across synthetic vault sizes.

Usage:
    python scripts/benchmark.py              # sizes 100 1000 5000 10000
    python scripts/benchmark.py 100 2000     # custom sizes

Notes are generated deterministically (fixed random seed) from a vocabulary of
real words plus [[wiki-links]] and #tags, so runs are repeatable. Query latency
is the median over all queries x repeats, measured on the in-memory index
(the same SearchEngine the CLI and API use), including snippet extraction.
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from brain.index import Index
from brain.search import SearchEngine

WORDS = (
    "algorithm attention backlink cache compression context cosine dataset decay "
    "document embedding entropy evaluation feature frequency gradient graph heuristic "
    "index inverse kernel knowledge latent learning linear matrix memory metric model "
    "network neural node normalization optimization orphan parser pipeline precision "
    "query ranking recall regularization relevance retrieval scheduler search semantic "
    "similarity snippet sparse spectrum stemming summary tensor token topology "
    "transformer traversal vector vocabulary weight zettelkasten"
).split()
TAGS = ["ml", "nlp", "search", "pkm", "graphs", "math", "productivity", "systems"]
QUERIES = [
    "semantic search ranking",
    "neural network gradient",
    "knowledge graph traversal",
    "memory decay scheduler",
    "sparse vector similarity",
    "token stemming parser",
]
REPEATS = 5


def generate_vault(directory: Path, n_notes: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    for i in range(n_notes):
        title_words = rng.sample(WORDS, 3)
        body = " ".join(rng.choices(WORDS, k=rng.randint(150, 350)))
        links = " ".join(f"[[note-{rng.randrange(n_notes)}]]" for _ in range(rng.randint(1, 4)))
        tags = rng.sample(TAGS, 2)
        text = (
            f"---\ntitle: {' '.join(title_words).title()}\ntags: [{', '.join(tags)}]\n---\n"
            f"# {' '.join(title_words).title()}\n\n{body}\n\nSee also {links}\n"
        )
        (directory / f"note-{i}.md").write_text(text, encoding="utf-8")


def bench(n_notes: int) -> tuple[float, float, float]:
    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp)
        generate_vault(vault, n_notes)

        t0 = time.perf_counter()
        index = Index(vault)
        index.build(force=True)
        index_secs = time.perf_counter() - t0

        engine = SearchEngine(index)
        engine.search(QUERIES[0], k=10)  # warm-up (file cache)

        samples: list[float] = []
        for _ in range(REPEATS):
            for q in QUERIES:
                t = time.perf_counter()
                engine.search(q, k=10)
                samples.append((time.perf_counter() - t) * 1000)

        return index_secs, statistics.median(samples), max(samples)


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [100, 1000, 5000, 10000]
    print(f"{'notes':>8} | {'index build':>12} | {'median query':>12} | {'max query':>10}")
    print("-" * 52)
    for n in sizes:
        index_secs, median_ms, max_ms = bench(n)
        print(f"{n:>8} | {index_secs:>10.2f} s | {median_ms:>9.1f} ms | {max_ms:>7.1f} ms")


if __name__ == "__main__":
    main()
