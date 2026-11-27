"""Unit tests for knowledge graph engine: backlinks, outlinks, orphans, broken links, BFS."""

from pathlib import Path
import pytest
from brain.index import Index
from brain.graph import KnowledgeGraph


@pytest.fixture(scope="module")
def sample_graph() -> KnowledgeGraph:
    """Build knowledge graph over sample_notes."""
    sample_dir = Path("sample_notes").resolve()
    index = Index(sample_dir)
    index.build(force=True)
    return KnowledgeGraph(index)


def test_graph_node_and_edge_counts(sample_graph: KnowledgeGraph):
    stats = sample_graph.get_stats()
    assert stats.node_count == 20
    assert stats.edge_count >= 30


def test_orphan_note_detection(sample_graph: KnowledgeGraph):
    """Verify that isolated-orphan-concept.md is detected as an orphan note."""
    orphans = sample_graph.get_orphans()
    assert "isolated-orphan-concept.md" in orphans


def test_broken_link_detection(sample_graph: KnowledgeGraph):
    """Verify that intentional broken link [[non-existent-architecture]] is detected."""
    stats = sample_graph.get_stats()
    broken_targets = [target for source, target in stats.broken_links]
    assert "non-existent-architecture" in broken_targets


def test_bidirectional_links(sample_graph: KnowledgeGraph):
    """Test that links and backlinks correspond correctly."""
    # attention-mechanisms links to transformers
    outlinks = sample_graph.get_outlinks("attention-mechanisms")
    assert "transformers.md" in outlinks

    # transformers should have backlink from attention-mechanisms
    backlinks = sample_graph.get_backlinks("transformers")
    assert "attention-mechanisms.md" in backlinks


def test_most_connected_notes(sample_graph: KnowledgeGraph):
    most_conn = sample_graph.get_most_connected(limit=5)
    assert len(most_conn) == 5
    # Degree should be descending
    degrees = [item[1] for item in most_conn]
    assert degrees == sorted(degrees, reverse=True)


def test_bfs_neighbors(sample_graph: KnowledgeGraph):
    # Depth 1 from transformers
    d1 = sample_graph.neighbors("transformers", depth=1)
    assert "attention-mechanisms.md" in d1
    assert d1["attention-mechanisms.md"] == 1

    # Depth 2 should find notes reachable in 2 hops
    d2 = sample_graph.neighbors("transformers", depth=2)
    assert len(d2) > len(d1)
    for path, dist in d2.items():
        assert dist in (1, 2)


def test_to_network_data_structure(sample_graph: KnowledgeGraph):
    net_data = sample_graph.to_network_data()
    assert "nodes" in net_data
    assert "edges" in net_data
    assert len(net_data["nodes"]) == 20
    assert len(net_data["edges"]) >= 30
    assert "id" in net_data["nodes"][0]
    assert "title" in net_data["nodes"][0]
    assert "degree" in net_data["nodes"][0]


def test_bfs_direction(sample_graph: KnowledgeGraph):
    out_only = sample_graph.neighbors("transformers", depth=1, direction="out")
    in_only = sample_graph.neighbors("transformers", depth=1, direction="in")
    both = sample_graph.neighbors("transformers", depth=1, direction="both")
    assert set(out_only) == set(sample_graph.get_outlinks("transformers"))
    assert set(in_only) == set(sample_graph.get_backlinks("transformers"))
    assert set(both) == set(out_only) | set(in_only)
    # Legacy flag still means outgoing-only
    assert sample_graph.neighbors("transformers", depth=1, directed=True) == out_only
