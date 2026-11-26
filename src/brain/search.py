"""TF-IDF vector space search, snippet extraction, and optional LSA semantic search."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.index import Index, IndexedDoc
from brain.tokenizer import stem_word, tokenize


@dataclass
class SearchResult:
    """Ranked search result item."""

    path: str
    title: str
    score: float
    snippet: str
    tags: list[str] = field(default_factory=list)
    highlights: list[tuple[int, int]] = field(default_factory=list)  # (start, end) offsets in snippet


def extract_snippet(
    content: str,
    query_stems: set[str],
    raw_query_words: set[str],
    window_size: int = 150,
) -> tuple[str, list[tuple[int, int]]]:
    """Find the densest ~150 character window containing query terms and mark match spans.

    Returns:
        (snippet_text, highlight_spans_in_snippet)
    """
    if not content:
        return "", []

    # Clean markdown formatting for readability
    clean = re.sub(r"```[\s\S]*?```", " ", content)
    clean = re.sub(r"^---[\s\S]*?---\s*", "", clean)  # strip frontmatter
    clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)  # strip # from headings
    clean = re.sub(r"\[\[(.*?)\]\]", r"\1", clean)  # unwrap [[link]]
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return "", []

    # Find word boundaries and match against query
    matches: list[tuple[int, int]] = []
    word_pattern = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b")

    for m in word_pattern.finditer(clean):
        word = m.group(0).lower()
        if word in raw_query_words or stem_word(word) in query_stems:
            matches.append((m.start(), m.end()))

    if not matches:
        # Fallback to first ~window_size chars
        snippet = clean[:window_size].strip()
        if len(clean) > window_size:
            snippet += "..."
        return snippet, []

    # Find the window containing maximum number of unique terms and total matches
    best_start = 0
    best_end = min(len(clean), window_size)
    best_score = -1

    for start_match_idx, (m_start, _) in enumerate(matches):
        target_end = m_start + window_size
        # Collect matches in [m_start, target_end]
        window_matches: list[tuple[int, int]] = []
        unique_stems: set[str] = set()

        for s, e in matches[start_match_idx:]:
            if s > target_end:
                break
            window_matches.append((s, e))
            w = clean[s:e].lower()
            unique_stems.add(stem_word(w))

        # Density score: unique terms weighted 100 + total matches
        score = len(unique_stems) * 100 + len(window_matches)
        if score > best_score:
            best_score = score
            best_start = m_start
            best_end = min(len(clean), target_end)

    # Expand boundaries slightly to nearest space
    actual_start = best_start
    if actual_start > 0:
        prev_space = clean.rfind(" ", max(0, actual_start - 20), actual_start)
        if prev_space != -1:
            actual_start = prev_space + 1

    actual_end = best_end
    if actual_end < len(clean):
        next_space = clean.find(" ", actual_end, min(len(clean), actual_end + 20))
        if next_space != -1:
            actual_end = next_space

    prefix = "..." if actual_start > 0 else ""
    suffix = "..." if actual_end < len(clean) else ""
    raw_snippet = clean[actual_start:actual_end]
    full_snippet = prefix + raw_snippet + suffix

    # Calculate highlight spans relative to full_snippet
    prefix_len = len(prefix)
    highlights: list[tuple[int, int]] = []
    for s, e in matches:
        if s >= actual_start and e <= actual_end:
            h_start = prefix_len + (s - actual_start)
            h_end = prefix_len + (e - actual_start)
            highlights.append((h_start, h_end))

    return full_snippet, highlights


class SearchEngine:
    """Ranked search engine supporting TF-IDF cosine similarity and optional LSA."""

    def __init__(self, index: Index) -> None:
        self.index = index

    def _build_query_vector(self, query_tokens: list[str]) -> tuple[dict[str, float], float]:
        """Build normalized TF-IDF vector for query tokens."""
        counts = Counter(query_tokens)
        weights: dict[str, float] = {}
        sum_sq = 0.0

        for term, count in counts.items():
            if term in self.index.idfs:
                tf = 1.0 + math.log(count)
                idf = self.index.idfs[term]
                w = tf * idf
                weights[term] = w
                sum_sq += w * w

        norm = math.sqrt(sum_sq)
        if norm == 0.0:
            return {}, 0.0

        normalized = {term: w / norm for term, w in weights.items()}
        return normalized, norm

    def search_keyword(
        self,
        query: str,
        k: int = 10,
        tag: str | None = None,
    ) -> list[SearchResult]:
        """Ranked cosine similarity search over TF-IDF vectors."""
        if not query.strip() or self.index.total_docs == 0:
            return []

        raw_query_words = set(re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b", query.lower()))
        query_stems_list = tokenize(query, stem=True)
        query_stems_set = set(query_stems_list)

        q_vec, q_norm = self._build_query_vector(query_stems_list)
        if q_norm == 0.0:
            return []

        # Accumulate dot product scores using inverted index
        scores: dict[str, float] = defaultdict(float)
        for term, q_weight in q_vec.items():
            postings = self.index.inverted_index.get(term, {})
            for doc_path, doc_weight in postings.items():
                scores[doc_path] += q_weight * doc_weight

        # Filter by tag if requested
        tag_lower = tag.lower().lstrip("#") if tag else None

        results: list[SearchResult] = []
        for doc_path, score in scores.items():
            if score <= 0.0:
                continue
            doc = self.index.docs.get(doc_path)
            if not doc:
                continue
            if tag_lower and tag_lower not in [t.lower() for t in doc.tags]:
                continue

            # Load document content for snippet extraction
            full_file = self.index.base_dir / doc_path
            content = ""
            if full_file.exists():
                try:
                    content = full_file.read_text(encoding="utf-8")
                except Exception:
                    content = ""

            snippet, highlights = extract_snippet(
                content=content,
                query_stems=query_stems_set,
                raw_query_words=raw_query_words,
            )

            results.append(
                SearchResult(
                    path=doc_path,
                    title=doc.title,
                    score=score,
                    snippet=snippet,
                    tags=doc.tags,
                    highlights=highlights,
                )
            )

        # Sort by score desc, then by path asc for deterministic tie-breaking
        results.sort(key=lambda r: (-r.score, r.path))
        return results[:k]

    def _get_lsa_matrices(self) -> tuple[Any, Any, Any, list[str], list[str]]:
        """Compute or retrieve cached SVD matrices for Latent Semantic Analysis.

        Returns:
            (Uk, inv_Sigma_k, doc_vectors_k, vocab_list, doc_paths)
        """
        try:
            import numpy as np
        except ImportError as err:
            raise ImportError("numpy is required for semantic search: pip install numpy") from err

        if self.index.svd_cache is not None:
            cache = self.index.svd_cache
            return (
                cache["Uk"],
                cache["inv_Sigma_k"],
                cache["doc_vectors_k"],
                cache["vocab"],
                cache["doc_paths"],
            )

        doc_paths = sorted(self.index.docs.keys())
        vocab = sorted(self.index.doc_freqs.keys())
        m = len(vocab)
        n = len(doc_paths)

        if m == 0 or n == 0:
            raise ValueError("Index vocabulary or document set is empty")

        vocab_index = {term: i for i, term in enumerate(vocab)}
        A = np.zeros((m, n), dtype=np.float64)

        for j, p in enumerate(doc_paths):
            doc = self.index.docs[p]
            for term, weight in doc.weights.items():
                if term in vocab_index:
                    A[vocab_index[term], j] = weight

        # SVD: A = U * Sigma * V^T
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        k = min(15, m, n)

        Uk = U[:, :k]  # (m, k)
        sk = s[:k]  # (k,)
        # Filter near-zero singular values for numerical stability
        inv_sk = np.where(sk > 1e-9, 1.0 / sk, 0.0)

        # Document representations in latent space: columns of Vt[:k, :]
        doc_vectors_k = Vt[:k, :]  # (k, n)
        # Normalize document vectors in latent space
        norms = np.linalg.norm(doc_vectors_k, axis=0, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        doc_vectors_k = doc_vectors_k / norms

        self.index.svd_cache = {
            "Uk": Uk,
            "inv_Sigma_k": inv_sk,
            "doc_vectors_k": doc_vectors_k,
            "vocab": vocab,
            "doc_paths": doc_paths,
        }

        return Uk, inv_sk, doc_vectors_k, vocab, doc_paths

    def search_semantic(
        self,
        query: str,
        k: int = 10,
        tag: str | None = None,
    ) -> list[SearchResult]:
        """Latent Semantic Analysis search (LSA) projecting query via q_k = Sigma^-1 * U^T * q."""
        try:
            import numpy as np
        except ImportError as err:
            raise ImportError("numpy is required for semantic search: pip install numpy") from err

        if not query.strip() or self.index.total_docs == 0:
            return []

        Uk, inv_sk, doc_vectors_k, vocab, doc_paths = self._get_lsa_matrices()
        vocab_index = {term: i for i, term in enumerate(vocab)}

        query_stems_list = tokenize(query, stem=True)
        q_vec, q_norm = self._build_query_vector(query_stems_list)
        if q_norm == 0.0:
            return []

        # Vector q in vocabulary space
        q = np.zeros(len(vocab), dtype=np.float64)
        for term, w in q_vec.items():
            if term in vocab_index:
                q[vocab_index[term]] = w

        # Query projection (folding-in): q_k = Sigma^-1 * U^T * q
        qk = inv_sk * (Uk.T @ q)  # element-wise multiply with Sigma^-1
        qk_norm = np.linalg.norm(qk)
        if qk_norm > 1e-9:
            qk = qk / qk_norm
        else:
            return []

        # Cosine similarity in latent space: qk . d_j
        scores_arr = qk @ doc_vectors_k  # shape: (n,)

        raw_query_words = set(re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b", query.lower()))
        query_stems_set = set(query_stems_list)
        tag_lower = tag.lower().lstrip("#") if tag else None

        results: list[SearchResult] = []
        for j, score in enumerate(scores_arr):
            score_val = float(score)
            if score_val <= 0.0:
                continue
            doc_path = doc_paths[j]
            doc = self.index.docs.get(doc_path)
            if not doc:
                continue
            if tag_lower and tag_lower not in [t.lower() for t in doc.tags]:
                continue

            full_file = self.index.base_dir / doc_path
            content = ""
            if full_file.exists():
                try:
                    content = full_file.read_text(encoding="utf-8")
                except Exception:
                    content = ""

            snippet, highlights = extract_snippet(
                content=content,
                query_stems=query_stems_set,
                raw_query_words=raw_query_words,
            )

            results.append(
                SearchResult(
                    path=doc_path,
                    title=doc.title,
                    score=score_val,
                    snippet=snippet,
                    tags=doc.tags,
                    highlights=highlights,
                )
            )

        results.sort(key=lambda r: (-r.score, r.path))
        return results[:k]

    def search(
        self,
        query: str,
        k: int = 10,
        tag: str | None = None,
        semantic: bool = False,
    ) -> list[SearchResult]:
        """Dispatch search to keyword or semantic mode."""
        if semantic:
            return self.search_semantic(query, k=k, tag=tag)
        return self.search_keyword(query, k=k, tag=tag)
