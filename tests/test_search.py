"""Unit tests for search ranking, snippet extraction, tag filtering, and optional LSA."""

from pathlib import Path
import pytest
from brain.index import Index
from brain.search import SearchEngine, extract_snippet


@pytest.fixture(scope="module")
def sample_engine() -> SearchEngine:
    sample_dir = Path("sample_notes").resolve()
    index = Index(sample_dir)
    index.build(force=True)
    return SearchEngine(index)


def test_keyword_search_ranking(sample_engine: SearchEngine):
    results = sample_engine.search("attention mechanism transformer", k=5)
    assert len(results) > 0
    # Top results should be the most relevant notes
    top_paths = [r.path for r in results[:2]]
    assert "attention-mechanisms.md" in top_paths or "transformers.md" in top_paths
    assert results[0].score > 0.0


def test_tag_filtering(sample_engine: SearchEngine):
    # Search across all notes
    all_results = sample_engine.search("learning", k=10)
    # Search with tag filter
    filtered = sample_engine.search("learning", k=10, tag="nlp")

    assert len(filtered) > 0
    for r in filtered:
        assert any(t.lower() == "nlp" for t in r.tags)


def test_snippet_extraction_densest_window():
    text = (
        "In artificial intelligence, neural networks learn complex representations. "
        "The transformer architecture relies on self-attention mechanisms to capture "
        "long range token dependencies in sequential data. Attention is essential."
    )
    snippet, highlights = extract_snippet(
        content=text,
        query_stems={"attent", "transform"},
        raw_query_words={"attention", "transformer"},
        window_size=120,
    )

    assert len(snippet) > 0
    assert "attention" in snippet.lower()
    assert "transformer" in snippet.lower()
    assert len(highlights) >= 2


def test_lsa_semantic_search(sample_engine: SearchEngine):
    """Test LSA semantic search with SVD caching and query folding."""
    results = sample_engine.search("neural architecture and deep modeling", k=5, semantic=True)
    assert len(results) > 0
    assert results[0].score > 0.0
    # Verify SVD was cached
    assert sample_engine.index.svd_cache is not None
    assert "Uk" in sample_engine.index.svd_cache
    assert "inv_Sigma_k" in sample_engine.index.svd_cache


def test_incremental_reindex_cache_invalidation(tmp_path: Path):
    """Test that modifying index invalidates SVD cache."""
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test\nSample note.", encoding="utf-8")

    index = Index(tmp_path)
    index.build()

    engine = SearchEngine(index)
    engine.search("sample", semantic=True)
    assert index.svd_cache is not None

    # Modify file and recompute
    test_file.write_text("# Test\nModified note with new concepts.", encoding="utf-8")
    index.build()
    # Cache should be invalidated (set to None)
    assert index.svd_cache is None


def test_keyword_search_works_without_numpy(monkeypatch, sample_engine: SearchEngine):
    """Verify keyword search functions completely without numpy (LSA stays optional)."""
    import sys
    # Simulate environment where numpy cannot be imported
    monkeypatch.setitem(sys.modules, "numpy", None)

    # Keyword search should succeed with zero dependency on numpy
    keyword_results = sample_engine.search("attention mechanism", semantic=False)
    assert len(keyword_results) > 0
    assert keyword_results[0].score > 0.0

    # Semantic search should raise clear informative error
    with pytest.raises(ImportError, match="numpy is required for semantic search"):
        sample_engine.search("attention mechanism", semantic=True)

