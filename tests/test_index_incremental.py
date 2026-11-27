"""Tests for incremental re-indexing parity against full rebuild."""

import shutil
from pathlib import Path
import pytest
from brain.index import Index
from brain.search import SearchEngine


def test_incremental_index_equals_fresh_rebuild(tmp_path: Path):
    """Build a full index of a temp copy of sample_notes, then modify one file,

    delete one, add one, re-index incrementally, and assert the result equals
    a fresh full rebuild (df, idf, weights, norms, search scores).
    """
    # 1. Copy sample_notes to a temporary directory with a space in path
    sample_src = Path("sample_notes").resolve()
    temp_dir = tmp_path / "temp sample notes"
    shutil.copytree(sample_src, temp_dir)

    # 2. Build initial index
    index_inc = Index(temp_dir)
    added_init, mod_init, del_init = index_inc.build(force=True)
    assert added_init == 20
    assert mod_init == 0
    assert del_init == 0
    assert index_inc.total_docs == 20

    # 3. Apply mutations:
    # (a) Modify one file: append content with new vocabulary to transformers.md
    mod_file = temp_dir / "transformers.md"
    assert mod_file.exists()
    original_mod_content = mod_file.read_text(encoding="utf-8")
    mod_file.write_text(
        original_mod_content + "\n\n## Future Horizons\nExploring synthetic token scaling and quantum transformers.\n",
        encoding="utf-8"
    )

    # (b) Delete one file: dynamic-programming.md
    del_file = temp_dir / "dynamic-programming.md"
    assert del_file.exists()
    del_file.unlink()

    # (c) Add one file: quantum-computing.md
    add_file = temp_dir / "quantum-computing.md"
    add_file.write_text(
        "---\ntitle: Quantum Computing Foundations\ntags: [quantum, physics, algorithms]\n---\n"
        "# Quantum Computing Foundations\n\nQuantum algorithms utilize qubits, superposition, and entanglement.\n",
        encoding="utf-8"
    )

    # 4. Incremental re-index on index_inc
    added_cnt, mod_cnt, del_cnt = index_inc.build(force=False)
    assert added_cnt == 1
    assert mod_cnt == 1
    assert del_cnt == 1
    assert index_inc.total_docs == 20  # 20 - 1 + 1 = 20

    # 5. Build fresh full index from scratch in a second Index instance
    fresh_dir = tmp_path / "fresh rebuild"
    # Copy the exact mutated directory state
    shutil.copytree(temp_dir, fresh_dir, ignore=shutil.ignore_patterns(".brain"))
    index_fresh = Index(fresh_dir)
    index_fresh.build(force=True)

    # 6. Assert parity between incremental and fresh rebuild
    # (a) Total documents & document keys
    assert index_inc.total_docs == index_fresh.total_docs
    assert set(index_inc.docs.keys()) == set(index_fresh.docs.keys())

    # (b) Document frequencies (df)
    assert index_inc.doc_freqs == index_fresh.doc_freqs

    # (c) Inverse Document Frequencies (idf)
    for term, fresh_idf in index_fresh.idfs.items():
        assert term in index_inc.idfs
        assert pytest.approx(index_inc.idfs[term], rel=1e-7) == fresh_idf

    # (d) Per-document L2 norms and normalized weights
    for doc_path, fresh_doc in index_fresh.docs.items():
        inc_doc = index_inc.docs[doc_path]
        assert pytest.approx(inc_doc.norm, rel=1e-7) == fresh_doc.norm
        assert set(inc_doc.weights.keys()) == set(fresh_doc.weights.keys())
        for term, fresh_w in fresh_doc.weights.items():
            assert pytest.approx(inc_doc.weights[term], rel=1e-7) == fresh_w

    # (e) Search scores parity across multiple queries
    engine_inc = SearchEngine(index_inc)
    engine_fresh = SearchEngine(index_fresh)

    test_queries = [
        "quantum transformers",
        "attention mechanism",
        "qubits entanglement",
        "regularization gradient",
    ]

    for q in test_queries:
        res_inc = engine_inc.search(q, k=10)
        res_fresh = engine_fresh.search(q, k=10)

        assert len(res_inc) == len(res_fresh)
        for r_inc, r_fresh in zip(res_inc, res_fresh):
            assert r_inc.path == r_fresh.path
            assert pytest.approx(r_inc.score, rel=1e-6) == r_fresh.score
