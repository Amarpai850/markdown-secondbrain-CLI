"""Command-line interface entry point for brain."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from brain.graph import KnowledgeGraph
from brain.highlight import (
    bold,
    cyan,
    dim,
    green,
    highlight_snippet,
    magenta,
    red,
    yellow,
)
from brain.index import Index
from brain.search import SearchEngine


class BrainCLIError(Exception):
    """User-facing CLI error displaying friendly message without traceback."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _configure_terminal_encoding() -> None:
    """Ensure standard output and error use UTF-8 on Windows consoles."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _get_target_dir(dir_arg: str | None) -> Path:
    """Resolve target directory from argument or current working directory."""
    target = Path(dir_arg).resolve() if dir_arg else Path.cwd().resolve()
    if not target.exists() or not target.is_dir():
        raise BrainCLIError(f"Directory '{target}' does not exist.")
    return target


def _load_index_or_err(base_dir: Path) -> Index:
    """Load index from directory or raise friendly error if not indexed yet."""
    index = Index(base_dir)
    if not index.index_file.exists():
        raise BrainCLIError(f"No index found at '{base_dir}'. Run 'brain index \"{base_dir}\"' first.")
    if not index.load():
        raise BrainCLIError(f"Index at '{base_dir}' is corrupted or unreadable. Run 'brain index --force' to rebuild.")
    return index


def cmd_index(args: argparse.Namespace) -> int:
    """Index a directory of markdown files."""
    base_dir = _get_target_dir(args.dir)
    index = Index(base_dir)

    print(f"Indexing markdown notes in {cyan(str(base_dir))}...")
    added, modified, deleted = index.build(force=args.force)

    total = index.total_docs
    status_parts = []
    if added > 0:
        status_parts.append(f"{green(str(added))} added")
    if modified > 0:
        status_parts.append(f"{yellow(str(modified))} modified")
    if deleted > 0:
        status_parts.append(f"{red(str(deleted))} deleted")

    changes_str = ", ".join(status_parts) if status_parts else "up to date"
    print(f"Indexed {bold(str(total))} documents ({changes_str}).")
    print(f"Index persisted to {dim(str(index.index_file))}.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Perform ranked keyword or semantic search."""
    query = args.query.strip()
    if not query:
        raise BrainCLIError("Search query cannot be empty.")

    base_dir = _get_target_dir(args.dir)
    index = _load_index_or_err(base_dir)
    engine = SearchEngine(index)

    mode_str = " (semantic LSA mode)" if args.semantic else ""
    tag_filter_str = f" [tag: #{args.tag}]" if args.tag else ""

    try:
        results = engine.search(
            query=query,
            k=args.k,
            tag=args.tag,
            semantic=args.semantic,
        )
    except ImportError as e:
        raise BrainCLIError(str(e))

    if not results:
        print(f"No matching notes found for '{query}'{mode_str}{tag_filter_str}.")
        return 0

    print(f"\nSearch results for '{bold(query)}'{dim(mode_str)}{dim(tag_filter_str)}:\n")

    for i, res in enumerate(results, 1):
        score_display = magenta(f"{res.score:.4f}")
        path_display = cyan(res.path)
        title_display = bold(res.title)
        tags_display = " ".join(dim(f"#{t}") for t in res.tags) if res.tags else ""

        print(f" {dim(str(i) + '.')} [{score_display}] {path_display} - {title_display} {tags_display}")

        if res.snippet:
            highlighted = highlight_snippet(res.snippet, res.highlights)
            print(f"    {dim('...') if not highlighted.startswith('...') else ''}{highlighted}")
        print()

    return 0


def cmd_links(args: argparse.Namespace) -> int:
    """Inspect outlinks and backlinks for a specific note."""
    base_dir = _get_target_dir(args.dir)
    index = _load_index_or_err(base_dir)
    graph = KnowledgeGraph(index)

    note_query = args.note.strip()
    resolved = graph.resolve_note(note_query)
    if not resolved:
        raise BrainCLIError(f"Note '{note_query}' not found in index.")

    canonical_path = graph.doc_map.get(resolved, resolved)
    doc = index.docs.get(canonical_path)
    title = doc.title if doc else canonical_path

    outlinks = graph.get_outlinks(note_query)
    backlinks = graph.get_backlinks(note_query)

    print(f"\nLinks for {bold(title)} ({cyan(canonical_path)}):\n")

    print(f" {bold('Outgoing Links')} ({len(outlinks)}):")
    if outlinks:
        for tgt in outlinks:
            tgt_doc = index.docs.get(tgt)
            tgt_title = tgt_doc.title if tgt_doc else tgt
            print(f"   -> {cyan(tgt)} {dim('(' + tgt_title + ')')}")
    else:
        print(f"   {dim('No outgoing links.')}")

    print(f"\n {bold('Backlinks')} ({len(backlinks)}):")
    if backlinks:
        for src in backlinks:
            src_doc = index.docs.get(src)
            src_title = src_doc.title if src_doc else src
            print(f"   <- {cyan(src)} {dim('(' + src_title + ')')}")
    else:
        print(f"   {dim('No incoming backlinks.')}")
    print()

    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Display knowledge graph summary statistics."""
    base_dir = _get_target_dir(args.dir)
    index = _load_index_or_err(base_dir)
    graph = KnowledgeGraph(index)

    stats = graph.get_stats()

    print(f"\n{bold('Knowledge Graph Statistics')} for {cyan(str(base_dir))}:\n")
    print(f"  {bold('Total Notes (Nodes):')}      {green(str(stats.node_count))}")
    print(f"  {bold('Total Connections (Edges):')} {green(str(stats.edge_count))}")
    print(f"  {bold('Orphan Notes:')}             {yellow(str(len(stats.orphan_notes)))}")
    print(f"  {bold('Broken Links:')}             {red(str(len(stats.broken_links)))}\n")

    if stats.orphan_notes:
        print(f" {bold('Orphan Notes')} (no links in or out):")
        for orphan in stats.orphan_notes:
            print(f"   - {yellow(orphan)}")
        print()

    if stats.broken_links:
        print(f" {bold('Broken Links')} (target does not exist):")
        for src, target in stats.broken_links:
            print(f"   - {cyan(src)} -> {red('[[' + target + ']]')}")
        print()

    print(f" {bold('Most Connected Notes')}:")
    for path, total, in_deg, out_deg in stats.most_connected[:5]:
        doc = index.docs.get(path)
        title = doc.title if doc else path
        print(f"   - {cyan(path)}: {bold(str(total))} links ({in_deg} in, {out_deg} out) {dim('- ' + title)}")
    print()

    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    """Find neighbor notes within N hops using BFS."""
    base_dir = _get_target_dir(args.dir)
    index = _load_index_or_err(base_dir)
    graph = KnowledgeGraph(index)

    note_query = args.note.strip()
    resolved = graph.resolve_note(note_query)
    if not resolved:
        raise BrainCLIError(f"Note '{note_query}' not found in index.")

    canonical_path = graph.doc_map.get(resolved, resolved)
    doc = index.docs.get(canonical_path)
    title = doc.title if doc else canonical_path

    depth = max(1, args.depth)
    neighbors_map = graph.neighbors(note_query, depth=depth)

    print(f"\nNeighbors of {bold(title)} ({cyan(canonical_path)}) within depth {depth}:\n")

    if not neighbors_map:
        print(f"  {dim('No connected neighbors found within depth ' + str(depth) + '.')}\n")
        return 0

    # Group by distance
    by_distance: dict[int, list[str]] = {}
    for path, dist in neighbors_map.items():
        by_distance.setdefault(dist, []).append(path)

    for dist in sorted(by_distance.keys()):
        print(f" {bold('Hop ' + str(dist))} ({len(by_distance[dist])} notes):")
        for path in sorted(by_distance[dist]):
            ndoc = index.docs.get(path)
            ntitle = ndoc.title if ndoc else path
            print(f"   - {cyan(path)} {dim('(' + ntitle + ')')}")
        print()

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start local web API server."""
    base_dir = _get_target_dir(args.dir)

    # Validate that web dependencies are available
    try:
        import fastapi
        import uvicorn
        from brain.api import create_app
    except ImportError:
        raise BrainCLIError(
            "Web extra dependencies not installed.\n"
            "Install with: pip install \"brain[web]\" or pip install fastapi uvicorn"
        )

    # Ensure index exists before serving
    _load_index_or_err(base_dir)

    host = args.host
    port = args.port

    print(f"Starting Brain Web UI on {cyan(f'http://{host}:{port}')} serving {dim(str(base_dir))}...")
    app = create_app(base_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="brain",
        description="Markdown 'Second Brain' local CLI, search engine, and knowledge graph.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # index
    p_index = subparsers.add_parser("index", help="Build or incrementally update note index")
    p_index.add_argument("dir", nargs="?", default=None, help="Directory of notes to index (defaults to cwd)")
    p_index.add_argument("--force", action="store_true", help="Force full rebuild of index from scratch")

    # search
    p_search = subparsers.add_parser("search", help="Ranked vector search over indexed notes")
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument("-k", type=int, default=10, help="Maximum number of results (default: 10)")
    p_search.add_argument("--tag", default=None, help="Filter results by tag")
    p_search.add_argument("--semantic", action="store_true", help="Enable Latent Semantic Analysis (LSA) search")
    p_search.add_argument("--dir", default=None, help="Base notes directory (defaults to cwd)")

    # links
    p_links = subparsers.add_parser("links", help="Display outgoing links and incoming backlinks")
    p_links.add_argument("note", help="Note filename, stem, or title")
    p_links.add_argument("--dir", default=None, help="Base notes directory (defaults to cwd)")

    # graph
    p_graph = subparsers.add_parser("graph", help="Knowledge graph statistics and topology")
    p_graph.add_argument("--stats", action="store_true", default=True, help="Show graph topology statistics")
    p_graph.add_argument("--dir", default=None, help="Base notes directory (defaults to cwd)")

    # neighbors
    p_neighbors = subparsers.add_parser("neighbors", help="Find connected neighbors within N hops using BFS")
    p_neighbors.add_argument("note", help="Starting note filename, stem, or title")
    p_neighbors.add_argument("--depth", type=int, default=2, help="BFS traversal depth (default: 2)")
    p_neighbors.add_argument("--dir", default=None, help="Base notes directory (defaults to cwd)")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the web UI and REST API server locally")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    p_serve.add_argument("--dir", default=None, help="Base notes directory (defaults to cwd)")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    _configure_terminal_encoding()
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            return cmd_index(args)
        elif args.command == "search":
            return cmd_search(args)
        elif args.command == "links":
            return cmd_links(args)
        elif args.command == "graph":
            return cmd_graph(args)
        elif args.command == "neighbors":
            return cmd_neighbors(args)
        elif args.command == "serve":
            return cmd_serve(args)
        else:
            parser.print_help()
            return 0
    except BrainCLIError as err:
        print(f"Error: {err}", file=sys.stderr)
        return err.exit_code
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130
    except Exception as err:
        print(f"Unexpected error: {err}", file=sys.stderr)
        return 1
