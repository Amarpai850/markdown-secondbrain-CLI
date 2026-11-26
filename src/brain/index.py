"""Inverted index, TF-IDF weights, persistence, and incremental re-indexing."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from brain.parser import ParsedNote, parse_markdown
from brain.scanner import FileMeta, diff_manifests, scan_directory
from brain.tokenizer import tokenize


@dataclass
class IndexedDoc:
    """Document metadata and term frequency weights stored in the index."""

    path: str  # POSIX relative path
    title: str  # Note title
    headings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    term_counts: dict[str, int] = field(default_factory=dict)  # Boosted raw term counts
    weights: dict[str, float] = field(default_factory=dict)  # L2-normalized TF-IDF weights
    norm: float = 0.0  # Pre-normalization L2 norm
    mtime: float = 0.0
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexedDoc:
        return cls(
            path=data["path"],
            title=data["title"],
            headings=list(data.get("headings", [])),
            tags=list(data.get("tags", [])),
            links=list(data.get("links", [])),
            term_counts=dict(data.get("term_counts", {})),
            weights=dict(data.get("weights", {})),
            norm=float(data.get("norm", 0.0)),
            mtime=float(data.get("mtime", 0.0)),
            size=int(data.get("size", 0)),
        )


class Index:
    """Inverted index managing TF-IDF document vectors and disk persistence."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.index_dir = self.base_dir / ".brain"
        self.index_file = self.index_dir / "index.json"

        # State
        self.docs: dict[str, IndexedDoc] = {}
        self.doc_freqs: dict[str, int] = {}
        self.idfs: dict[str, float] = {}
        self.inverted_index: dict[str, dict[str, float]] = defaultdict(dict)  # term -> {doc_path: weight}
        self.svd_cache: dict[str, Any] | None = None  # In-memory LSA cache

    @property
    def total_docs(self) -> int:
        return len(self.docs)

    def compute_doc_term_counts(self, note: ParsedNote) -> dict[str, int]:
        """Compute boosted term counts for a parsed note.

        Title terms: 3x boost.
        Heading terms: 3x boost.
        Body terms: 1x weight.
        Note: Headings were already stripped from body in parser to prevent double-counting.
        """
        counts: Counter[str] = Counter()

        # Body tokens (1x)
        if note.body:
            for term in tokenize(note.body, stem=True):
                counts[term] += 1

        # Heading tokens (3x boost)
        for h in note.headings:
            for term in tokenize(h, stem=True):
                counts[term] += 3

        # Title tokens (3x boost)
        if note.title:
            for term in tokenize(note.title, stem=True):
                counts[term] += 3

        return dict(counts)

    def recompute_weights(self) -> None:
        """Recompute DF, IDF, TF-IDF weights, and L2 norms for all documents.

        Called after any add, modify, or delete operation.
        Formulas:
          tf(t, d) = 1 + ln(count(t, d)) if count > 0 else 0
          idf(t, D) = ln((N + 1) / (df(t) + 1)) + 1
          w(t, d) = tf(t, d) * idf(t, D)
          L2 norm = sqrt(sum(w^2))
          normalized_weight = w / norm
        """
        n_docs = len(self.docs)
        self.doc_freqs.clear()
        self.idfs.clear()
        self.inverted_index.clear()
        self.svd_cache = None  # Invalidate LSA cache when index changes

        if n_docs == 0:
            return

        # 1. Compute Document Frequencies (df)
        df_counter: Counter[str] = Counter()
        for doc in self.docs.values():
            for term in doc.term_counts:
                df_counter[term] += 1
        self.doc_freqs = dict(df_counter)

        # 2. Compute Inverse Document Frequencies (idf)
        for term, df in self.doc_freqs.items():
            self.idfs[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

        # 3. Compute TF-IDF weights and L2 norms for each document
        for doc_path, doc in self.docs.items():
            raw_weights: dict[str, float] = {}
            sum_sq = 0.0

            for term, count in doc.term_counts.items():
                if count > 0:
                    tf = 1.0 + math.log(count)
                    idf = self.idfs.get(term, 1.0)
                    w = tf * idf
                    raw_weights[term] = w
                    sum_sq += w * w

            norm = math.sqrt(sum_sq)
            doc.norm = norm
            doc.weights = {}

            if norm > 0.0:
                for term, w in raw_weights.items():
                    normalized_w = w / norm
                    doc.weights[term] = normalized_w
                    self.inverted_index[term][doc_path] = normalized_w

    def add_or_update_doc(self, rel_path: str, meta: FileMeta) -> None:
        """Read and parse a markdown file, computing raw term counts."""
        full_path = self.base_dir / rel_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = full_path.read_text(encoding="utf-8", errors="replace")

        note = parse_markdown(content, rel_path=rel_path)
        term_counts = self.compute_doc_term_counts(note)

        doc = IndexedDoc(
            path=rel_path,
            title=note.title,
            headings=note.headings,
            tags=note.tags,
            links=note.links,
            term_counts=term_counts,
            mtime=meta.mtime,
            size=meta.size,
        )
        self.docs[rel_path] = doc

    def remove_doc(self, rel_path: str) -> None:
        """Remove a document from the index."""
        if rel_path in self.docs:
            del self.docs[rel_path]

    def build(self, force: bool = False) -> tuple[int, int, int]:
        """Perform full or incremental build.

        Returns:
            (added_count, modified_count, deleted_count)
        """
        current_manifest = scan_directory(self.base_dir)

        if force:
            self.docs.clear()
            for rel_path, meta in current_manifest.items():
                self.add_or_update_doc(rel_path, meta)
            self.recompute_weights()
            self.save()
            return len(current_manifest), 0, 0

        # Incremental update based on disk manifest
        old_manifest = {
            doc.path: FileMeta(doc.path, doc.mtime, doc.size)
            for doc in self.docs.values()
        }
        added, modified, deleted = diff_manifests(old_manifest, current_manifest)

        if not added and not modified and not deleted and self.docs:
            return 0, 0, 0

        for rel_path in deleted:
            self.remove_doc(rel_path)

        for rel_path in added | modified:
            self.add_or_update_doc(rel_path, current_manifest[rel_path])

        self.recompute_weights()
        self.save()
        return len(added), len(modified), len(deleted)

    def save(self) -> None:
        """Persist index state atomically as JSON to .brain/index.json."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "base_dir": str(self.base_dir),
            "docs": {path: doc.to_dict() for path, doc in self.docs.items()},
            "doc_freqs": self.doc_freqs,
            "idfs": self.idfs,
        }
        temp_file = self.index_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.index_file)

    def load(self) -> bool:
        """Load index from .brain/index.json if it exists.

        Returns True if loaded successfully, False otherwise.
        """
        if not self.index_file.exists():
            return False

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.docs = {
                path: IndexedDoc.from_dict(d)
                for path, d in data.get("docs", {}).items()
            }
            self.doc_freqs = {
                k: int(v) for k, v in data.get("doc_freqs", {}).items()
            }
            self.idfs = {
                k: float(v) for k, v in data.get("idfs", {}).items()
            }

            # Reconstruct in-memory inverted index mapping
            self.inverted_index.clear()
            for doc_path, doc in self.docs.items():
                for term, weight in doc.weights.items():
                    self.inverted_index[term][doc_path] = weight

            return True
        except Exception:
            return False
