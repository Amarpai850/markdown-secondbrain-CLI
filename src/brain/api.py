"""FastAPI web layer wrapping the core brain search and graph engine."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from brain.graph import KnowledgeGraph
from brain.index import Index
from brain.search import SearchEngine


def get_web_dir() -> Path:
    """Locate web directory reliably across editable and installed packages."""
    if "BRAIN_WEB_DIR" in os.environ:
        p = Path(os.environ["BRAIN_WEB_DIR"]).resolve()
        if (p / "index.html").exists():
            return p

    # Check relative to this module in src/brain
    pkg_web = Path(__file__).resolve().parent.parent.parent / "web"
    if (pkg_web / "index.html").exists():
        return pkg_web

    # Fallback to current working directory
    cwd_web = Path.cwd().resolve() / "web"
    if (cwd_web / "index.html").exists():
        return cwd_web

    return pkg_web


def compute_inline_script_sha256(html_path: Path) -> str:
    """Extract inline script from index.html and compute its base64 SHA-256 digest for CSP."""
    if not html_path.exists():
        return ""
    try:
        raw = html_path.read_bytes()
        start_tag = b"<script>"
        end_tag = b"</script>"
        start_idx = raw.find(start_tag)
        end_idx = raw.find(end_tag, start_idx)
        if start_idx != -1 and end_idx != -1:
            script_bytes = raw[start_idx + len(start_tag):end_idx]
            digest = hashlib.sha256(script_bytes).digest()
            b64 = base64.b64encode(digest).decode("ascii")
            return f"'sha256-{b64}'"
    except Exception:
        pass
    return ""


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject Content-Security-Policy with sha256 script hash and X-Content-Type-Options: nosniff."""

    def __init__(self, app: Any, script_hash: str = "") -> None:
        super().__init__(app)
        self.script_hash = script_hash

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        script_src = f"'self' {self.script_hash}" if self.script_hash else "'self' 'unsafe-inline'"
        response.headers["Content-Security-Policy"] = (
            f"default-src 'self'; "
            f"script-src {script_src}; "
            f"style-src 'self' 'unsafe-inline'; "
            f"img-src 'self' data:; "
            f"connect-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding window rate limiter: maximum 60 requests per minute per IP.

    Resistant to spoofed X-Forwarded-For headers behind reverse proxies (taking
    the rightmost client-connecting IP appended by Google Front End/Cloud Run),
    and automatically evicts expired entries so memory cannot grow without bound.
    """

    def __init__(
        self,
        app: Any,
        max_requests: int = 60,
        window_seconds: int = 60,
        max_tracked_ips: int = 10000,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_ips = max_tracked_ips
        self.ip_timestamps: dict[str, deque[float]] = defaultdict(deque)
        self.last_cleanup = time.time()

    def _cleanup_stale_entries(self, now: float) -> None:
        """Purge inactive IPs whose timestamps are outside the active window."""
        cutoff = now - self.window_seconds
        stale_keys = [
            ip for ip, timestamps in self.ip_timestamps.items()
            if not timestamps or timestamps[-1] <= cutoff
        ]
        for ip in stale_keys:
            self.ip_timestamps.pop(ip, None)

        # Hard ceiling: if tracked entries exceed capacity, evict oldest
        if len(self.ip_timestamps) > self.max_tracked_ips:
            excess = len(self.ip_timestamps) - self.max_tracked_ips
            keys_to_drop = list(self.ip_timestamps.keys())[:excess]
            for ip in keys_to_drop:
                self.ip_timestamps.pop(ip, None)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Behind Cloud Run / reverse proxies, Google Front End appends the real
        # client IP to the end of the X-Forwarded-For chain. Taking the rightmost
        # entry ensures client-spoofed prefix IPs (e.g. "X-Forwarded-For: 1.1.1.1")
        # cannot bypass the rate limiter.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ips = [item.strip() for item in forwarded.split(",") if item.strip()]
            client_ip = ips[-1] if ips else "127.0.0.1"
        elif request.client and request.client.host:
            client_ip = request.client.host
        else:
            client_ip = "127.0.0.1"

        now = time.time()

        # Periodic cleanup of expired IP records to strictly bound memory
        if now - self.last_cleanup > self.window_seconds or len(self.ip_timestamps) > 1000:
            self._cleanup_stale_entries(now)
            self.last_cleanup = now

        q = self.ip_timestamps[client_ip]

        # Evict timestamps outside active sliding window for current IP
        while q and q[0] <= now - self.window_seconds:
            q.popleft()

        if len(q) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded: maximum 60 requests per minute."},
                headers={"Retry-After": "60"},
            )

        q.append(now)
        return await call_next(request)


def create_app(notes_dir: Path | str | None = None) -> FastAPI:
    """Create and configure FastAPI application with lifespan index loading."""
    resolved_notes_dir: Path
    if notes_dir:
        resolved_notes_dir = Path(notes_dir).resolve()
    else:
        env_dir = os.environ.get("BRAIN_NOTES_DIR", "sample_notes")
        resolved_notes_dir = Path(env_dir).resolve()

    web_dir = get_web_dir()
    index_html = web_dir / "index.html"
    script_hash = compute_inline_script_sha256(index_html)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Load index once at startup as read-only; do not re-index at request time
        index = Index(resolved_notes_dir)
        if index.index_file.exists():
            index.load()
        else:
            index.build()

        app.state.index = index
        app.state.search_engine = SearchEngine(index)
        app.state.graph = KnowledgeGraph(index)
        yield

    app = FastAPI(
        title="Brain Second Brain Search & Knowledge Graph",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Rate limiter middleware (60 req/min)
    app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware, script_hash=script_hash)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """Handle browser default favicon request cleanly with 204 No Content."""
        return Response(status_code=204)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint indicating indexed note count and numpy availability."""
        has_numpy = False
        try:
            import numpy  # noqa: F401
            has_numpy = True
        except ImportError:
            has_numpy = False

        total_notes = app.state.index.total_docs if hasattr(app.state, "index") else 0
        return {
            "status": "ok",
            "indexed_notes": total_notes,
            "has_numpy": has_numpy,
        }

    @app.get("/api/search")
    async def search(
        q: str = Query("", description="Query string between 1 and 200 characters"),
        k: int = Query(10, description="Number of results, clamped to 1..50"),
        tag: str | None = Query(None, description="Optional tag filter"),
        semantic: bool = Query(False, description="Enable Latent Semantic Analysis (LSA) search"),
    ) -> Any:
        """Search indexed markdown notes by keyword TF-IDF or LSA semantic similarity."""
        query_clean = q.strip()
        if len(query_clean) < 1 or len(query_clean) > 200:
            return JSONResponse(
                status_code=400,
                content={"error": "Search query must be between 1 and 200 characters."},
            )

        clamped_k = max(1, min(50, k))

        if semantic:
            try:
                import numpy  # noqa: F401
            except ImportError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Semantic search requires numpy, which is not installed on the server."},
                )

        try:
            results = app.state.search_engine.search(
                query=query_clean,
                k=clamped_k,
                tag=tag,
                semantic=semantic,
            )
        except ImportError as err:
            return JSONResponse(status_code=400, content={"error": str(err)})

        return {
            "query": query_clean,
            "k": clamped_k,
            "tag": tag,
            "semantic": semantic,
            "total_results": len(results),
            "results": [
                {
                    "path": r.path,
                    "title": r.title,
                    "score": round(r.score, 4),
                    "snippet": r.snippet,
                    "tags": r.tags,
                    "highlights": r.highlights,
                }
                for r in results
            ],
        }

    @app.get("/api/note/{name:path}")
    async def get_note(name: str) -> Any:
        """Get note metadata and content resolving ONLY through the index, preventing path traversal."""
        graph: KnowledgeGraph = app.state.graph
        index: Index = app.state.index

        canonical_key = graph.resolve_note(name)
        if not canonical_key or canonical_key not in graph.doc_map:
            return JSONResponse(status_code=404, content={"error": f"Note '{name}' not found"})

        rel_path = graph.doc_map[canonical_key]
        doc = index.docs.get(rel_path)
        if not doc:
            return JSONResponse(status_code=404, content={"error": f"Note '{name}' not found"})

        full_path = index.base_dir / rel_path
        content = ""
        if full_path.exists():
            try:
                content = full_path.read_text(encoding="utf-8")
            except Exception:
                content = full_path.read_text(encoding="utf-8", errors="replace")

        outlinks = graph.get_outlinks(canonical_key)
        backlinks = graph.get_backlinks(canonical_key)

        return {
            "path": rel_path,
            "title": doc.title,
            "headings": doc.headings,
            "tags": doc.tags,
            "links": doc.links,
            "outlinks": outlinks,
            "backlinks": backlinks,
            "content": content,
        }

    @app.get("/api/graph")
    async def get_graph() -> Any:
        """Export full graph network data structure for UI visualization."""
        graph: KnowledgeGraph = app.state.graph
        stats = graph.get_stats()
        network_data = graph.to_network_data()

        network_data["orphan_notes"] = stats.orphan_notes
        network_data["broken_links"] = [
            {"source": src, "target": tgt} for src, tgt in stats.broken_links
        ]
        return network_data

    @app.get("/")
    async def serve_ui() -> Response:
        """Serve the single-page vanilla HTML/CSS/JS frontend."""
        web_d = get_web_dir()
        index_f = web_d / "index.html"
        if not index_f.exists():
            return JSONResponse(status_code=404, content={"error": "UI index.html not found"})
        return FileResponse(index_f, media_type="text/html")

    return app
