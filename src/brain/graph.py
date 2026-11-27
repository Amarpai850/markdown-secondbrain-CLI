"""Directed knowledge graph: backlinks, outlinks, orphans, broken links, and BFS neighbors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain.index import Index


@dataclass
class GraphStats:
    """Summary statistics for the knowledge graph."""

    node_count: int
    edge_count: int
    orphan_notes: list[str] = field(default_factory=list)
    broken_links: list[tuple[str, str]] = field(default_factory=list)  # (source, target)
    most_connected: list[tuple[str, int, int, int]] = field(default_factory=list)  # (note, total, in, out)


class KnowledgeGraph:
    """Directed graph representing note interconnections via [[wiki-links]]."""

    def __init__(self, index: Index) -> None:
        self.index = index

        # Graph representations
        self.nodes: set[str] = set()  # Canonical note keys (stems)
        self.doc_map: dict[str, str] = {}  # Canonical key -> rel_path
        self.title_map: dict[str, str] = {}  # Title lowercase -> canonical key

        self.out_edges: dict[str, set[str]] = defaultdict(set)  # source -> {targets}
        self.in_edges: dict[str, set[str]] = defaultdict(set)  # target -> {sources}
        self.broken: list[tuple[str, str]] = []  # (source, raw_target)

        self._build_graph()

    def _canonical_key(self, path_or_title: str) -> str:
        """Derive standard lookup key from path stem or title."""
        stem = Path(path_or_title).stem.lower().strip()
        return stem.replace(" ", "-")

    def _build_graph(self) -> None:
        """Construct graph nodes, directed edges, and identify broken links."""
        self.nodes.clear()
        self.doc_map.clear()
        self.title_map.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.broken.clear()

        # 1. Register all existing documents as nodes
        for rel_path, doc in self.index.docs.items():
            key = self._canonical_key(rel_path)
            self.nodes.add(key)
            self.doc_map[key] = rel_path
            # Also map clean title
            if doc.title:
                clean_title_key = self._canonical_key(doc.title)
                self.title_map[clean_title_key] = key

        # 2. Add edges for wiki-links
        for rel_path, doc in self.index.docs.items():
            source_key = self._canonical_key(rel_path)
            for raw_link in doc.links:
                target_key = self.resolve_note(raw_link)
                if target_key:
                    self.out_edges[source_key].add(target_key)
                    self.in_edges[target_key].add(source_key)
                else:
                    self.broken.append((rel_path, raw_link))

    def resolve_note(self, query_name: str) -> str | None:
        """Resolve a note reference (filename, stem, or title) to its canonical node key."""
        target_clean = self._canonical_key(query_name)

        if target_clean in self.nodes:
            return target_clean
        if target_clean in self.title_map:
            return self.title_map[target_clean]

        # Check filename with or without dashes
        no_dashes = target_clean.replace("-", "")
        for n in self.nodes:
            if n.replace("-", "") == no_dashes:
                return n

        return None

    def get_outlinks(self, note_query: str) -> list[str]:
        """Get list of notes linked from the specified note."""
        node = self.resolve_note(note_query)
        if not node:
            return []
        return sorted(self.doc_map.get(tgt, tgt) for tgt in self.out_edges.get(node, set()))

    def get_backlinks(self, note_query: str) -> list[str]:
        """Get list of notes linking to the specified note."""
        node = self.resolve_note(note_query)
        if not node:
            return []
        return sorted(self.doc_map.get(src, src) for src in self.in_edges.get(node, set()))

    def get_orphans(self) -> list[str]:
        """Get notes with zero incoming links and zero outgoing links."""
        orphans: list[str] = []
        for node in sorted(self.nodes):
            in_deg = len(self.in_edges.get(node, set()))
            out_deg = len(self.out_edges.get(node, set()))
            if in_deg == 0 and out_deg == 0:
                orphans.append(self.doc_map.get(node, node))
        return orphans

    def get_most_connected(self, limit: int = 10) -> list[tuple[str, int, int, int]]:
        """Get top notes ranked by total degree (in_degree + out_degree).

        Returns:
            list of (rel_path, total_degree, in_degree, out_degree)
        """
        degree_list: list[tuple[str, int, int, int]] = []
        for node in self.nodes:
            in_deg = len(self.in_edges.get(node, set()))
            out_deg = len(self.out_edges.get(node, set()))
            total = in_deg + out_deg
            path = self.doc_map.get(node, node)
            degree_list.append((path, total, in_deg, out_deg))

        # Sort by total desc, then in_deg desc, then path asc
        degree_list.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return degree_list[:limit]

    def neighbors(
        self,
        note_query: str,
        depth: int = 1,
        directed: bool = False,
        direction: str | None = None,
    ) -> dict[str, int]:
        """BFS traversal to find all notes within specified hop distance.

        direction: "both" (default) follows links and backlinks, "out" follows
        outgoing links only, "in" follows incoming backlinks only. The older
        ``directed=True`` flag is kept as an alias for direction="out".

        Returns:
            dict mapping resolved note path to hop distance (excluding starting note).
        """
        if direction is None:
            direction = "out" if directed else "both"
        if direction not in ("both", "out", "in"):
            raise ValueError(f"direction must be 'both', 'out' or 'in', got {direction!r}")

        start_node = self.resolve_note(note_query)
        if not start_node or depth < 1:
            return {}

        visited: dict[str, int] = {start_node: 0}
        queue: deque[tuple[str, int]] = deque([(start_node, 0)])

        while queue:
            curr, dist = queue.popleft()
            if dist >= depth:
                continue

            # Connected neighbors
            next_nodes: set[str] = set()
            if direction in ("both", "out"):
                next_nodes |= self.out_edges.get(curr, set())
            if direction in ("both", "in"):
                next_nodes |= self.in_edges.get(curr, set())

            for nxt in next_nodes:
                if nxt not in visited:
                    visited[nxt] = dist + 1
                    queue.append((nxt, dist + 1))

        # Convert canonical keys to paths, omitting the starting note
        result: dict[str, int] = {}
        for node_key, dist in visited.items():
            if node_key != start_node:
                path = self.doc_map.get(node_key, node_key)
                result[path] = dist

        return result

    def get_stats(self) -> GraphStats:
        """Compute aggregate graph statistics."""
        edge_count = sum(len(tgts) for tgts in self.out_edges.values())
        orphans = self.get_orphans()
        most_conn = self.get_most_connected()

        return GraphStats(
            node_count=len(self.nodes),
            edge_count=edge_count,
            orphan_notes=orphans,
            broken_links=self.broken,
            most_connected=most_conn,
        )

    def to_network_data(self) -> dict[str, Any]:
        """Export graph data structure formatted for web visualizers."""
        nodes_data: list[dict[str, Any]] = []
        for node in sorted(self.nodes):
            path = self.doc_map.get(node, node)
            doc = self.index.docs.get(path)
            title = doc.title if doc else node
            tags = doc.tags if doc else []
            in_deg = len(self.in_edges.get(node, set()))
            out_deg = len(self.out_edges.get(node, set()))
            nodes_data.append({
                "id": path,
                "title": title,
                "tags": tags,
                "degree": in_deg + out_deg,
                "in_degree": in_deg,
                "out_degree": out_deg,
            })

        edges_data: list[dict[str, str]] = []
        for src, tgts in sorted(self.out_edges.items()):
            src_path = self.doc_map.get(src, src)
            for tgt in sorted(tgts):
                tgt_path = self.doc_map.get(tgt, tgt)
                edges_data.append({
                    "source": src_path,
                    "target": tgt_path,
                })

        return {
            "nodes": nodes_data,
            "edges": edges_data,
        }
