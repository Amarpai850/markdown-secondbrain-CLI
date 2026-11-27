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



# --- Tag boosting -----------------------------------------------------------

def test_count_matching_tags_handles_hyphens_and_stems():
    from brain.search import count_matching_tags
    from brain.tokenizer import tokenize

    stems = set(tokenize("machine learning transformer", stem=True))
    tags = ["machine-learning", "transformers", "nlp"]
    # "machine-learning" matches both words, "transformers" matches by stem, "nlp" does not
    assert count_matching_tags(tags, stems) == 2
    # A multi-word tag needs every word in the query
    assert count_matching_tags(["machine-learning"], set(tokenize("machine", stem=True))) == 0
    assert count_matching_tags(tags, set()) == 0


def test_tag_boost_factor_is_capped():
    from brain.search import MAX_BOOSTED_TAGS, TAG_BOOST, tag_boost_factor
    from brain.tokenizer import tokenize

    stems = set(tokenize("alpha beta gamma delta", stem=True))
    assert tag_boost_factor([], stems) == 1.0
    assert tag_boost_factor(["alpha"], stems) == pytest.approx(1.0 + TAG_BOOST)
    many = tag_boost_factor(["alpha", "beta", "gamma", "delta"], stems)
    assert many == pytest.approx(1.0 + TAG_BOOST * MAX_BOOSTED_TAGS)


def test_tag_match_ranks_note_higher(tmp_path: Path):
    """Two notes with identical text: the one tagged with the query word ranks first."""
    body = "# Note\n\nGardening tips about soil and compost for beginners.\n"
    (tmp_path / "a-untagged.md").write_text(body, encoding="utf-8")
    (tmp_path / "b-tagged.md").write_text("---\ntags: [compost]\n---\n" + body, encoding="utf-8")
    (tmp_path / "c-other.md").write_text("# Other\n\nUnrelated text about rivers.\n", encoding="utf-8")
    index = Index(tmp_path)
    index.build(force=True)
    results = SearchEngine(index).search("compost", k=5)
    assert [r.path for r in results[:2]] == ["b-tagged.md", "a-untagged.md"]
    assert results[0].score > results[1].score


# --- LSA rank selection and edge cases --------------------------------------

def test_choose_lsa_rank_edge_cases():
    from brain.search import LSA_MAX_RANK, choose_lsa_rank

    # All singular values zero: nothing usable
    assert choose_lsa_rank([0.0, 0.0]) == 0
    # A single positive value can never give rank above 1
    assert choose_lsa_rank([5.0, 0.0, 0.0]) == 1
    # Dominant first value: energy reached at k=1 but clamped up to the minimum of 2
    assert choose_lsa_rank([10.0, 1.0, 1.0, 1.0]) == 2
    # Flat spectrum: needs 80% of 10 equal values -> 8
    assert choose_lsa_rank([1.0] * 10) == 8
    # Never above the max rank
    assert choose_lsa_rank([1.0] * 500) == LSA_MAX_RANK


def test_semantic_search_stopword_only_index_returns_empty(tmp_path: Path):
    pytest.importorskip("numpy")
    (tmp_path / "a.md").write_text("the and of to a\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("is it on in at\n", encoding="utf-8")
    index = Index(tmp_path)
    index.build(force=True)
    engine = SearchEngine(index)
    assert engine.search("anything", semantic=True) == []


def test_semantic_search_drops_noise_scores(sample_engine: SearchEngine):
    pytest.importorskip("numpy")
    from brain.search import MIN_SEMANTIC_SCORE

    results = sample_engine.search("neural representations", k=50, semantic=True)
    assert results
    assert all(r.score > MIN_SEMANTIC_SCORE for r in results)
