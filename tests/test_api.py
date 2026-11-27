"""API integration tests using FastAPI TestClient covering endpoints, validation, traversal, and XSS."""

import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from brain.api import create_app
from brain.index import Index


@pytest.fixture(scope="module")
def api_test_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary notes directory with a space in its path and an XSS test note."""
    base_tmp = tmp_path_factory.mktemp("api test notes")
    notes_dir = base_tmp / "my test notes"
    sample_src = Path("sample_notes").resolve()
    shutil.copytree(sample_src, notes_dir)

    # Add an XSS test note containing raw script tag
    xss_note = notes_dir / "xss-security-test.md"
    xss_note.write_text(
        "---\ntitle: XSS Security Test\ntags: [security, testing]\n---\n"
        "# XSS Security Test\n\n"
        "Here is an inert payload: <script>alert(1)</script> and <img src=x onerror=alert(2)>.\n",
        encoding="utf-8",
    )

    # Build the index in this directory
    idx = Index(notes_dir)
    idx.build(force=True)
    return notes_dir


@pytest.fixture(scope="module")
def client(api_test_dir: Path) -> TestClient:
    """Instantiate TestClient using lifespan-managed app."""
    app = create_app(api_test_dir)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_notes"] >= 18
    assert "has_numpy" in data


def test_security_headers(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "Content-Security-Policy" in resp.headers
    assert "X-Content-Type-Options" in resp.headers
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    csp = resp.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    # Script-src uses sha256 hash instead of unsafe-inline
    assert "sha256-" in csp or "'unsafe-inline'" in csp


def test_ui_index_endpoint(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("Content-Type", "")
    assert "<title>Brain" in resp.text
    assert "graphCanvas" in resp.text


def test_search_valid_query(client: TestClient):
    resp = client.get("/api/search?q=machine+learning&k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "machine learning"
    assert data["k"] == 5
    assert len(data["results"]) > 0
    top = data["results"][0]
    assert "path" in top
    assert "score" in top
    assert "snippet" in top
    assert "highlights" in top


def test_search_tag_filtering(client: TestClient):
    resp = client.get("/api/search?q=learning&tag=nlp")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tag"] == "nlp"
    for r in data["results"]:
        assert any(t.lower() == "nlp" for t in r["tags"])


def test_search_input_validation(client: TestClient):
    # Empty query -> 400
    r_empty = client.get("/api/search?q=")
    assert r_empty.status_code == 400
    assert "between 1 and 200" in r_empty.json()["error"]

    # Whitespace only -> 400
    r_spaces = client.get("/api/search?q=   ")
    assert r_spaces.status_code == 400
    assert "between 1 and 200" in r_spaces.json()["error"]

    # Over 200 characters -> 400
    r_long = client.get(f"/api/search?q={'a' * 201}")
    assert r_long.status_code == 400
    assert "between 1 and 200" in r_long.json()["error"]

    # Clamp k > 50 down to 50
    r_k_high = client.get("/api/search?q=learning&k=100")
    assert r_k_high.status_code == 200
    assert r_k_high.json()["k"] == 50

    # Clamp k < 1 up to 1
    r_k_low = client.get("/api/search?q=learning&k=-5")
    assert r_k_low.status_code == 200
    assert r_k_low.json()["k"] == 1


def test_semantic_search_missing_numpy_returns_4xx(client: TestClient, monkeypatch):
    """When numpy is not installed, semantic=true must return clear 4xx error, not 500."""
    import sys
    monkeypatch.setitem(sys.modules, "numpy", None)

    resp = client.get("/api/search?q=transformer&semantic=true")
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert "numpy" in data["error"].lower()


def test_get_note_success(client: TestClient):
    resp = client.get("/api/note/transformers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "transformers.md"
    assert "Transformer Architecture" in data["title"]
    assert "attention-mechanisms.md" in data["outlinks"]
    assert "content" in data


def test_get_note_unknown_returns_404(client: TestClient):
    resp = client.get("/api/note/non-existent-note-xyz")
    assert resp.status_code == 404
    err_msg = resp.json().get("error") or resp.json().get("detail", "")
    assert "not found" in err_msg.lower()


def test_path_traversal_attempts_return_404(client: TestClient):
    """Verify that path traversal attempts resolve through index only and return 404."""
    traversal_payloads = [
        "../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "../../pyproject.toml",
        "..%2f..%2fpyproject.toml",
        "../../../Windows/win.ini",
        "....//....//etc/shadow",
    ]
    for payload in traversal_payloads:
        resp = client.get(f"/api/note/{payload}")
        assert resp.status_code == 404, f"Payload {payload} did not return 404"
        err_msg = resp.json().get("error") or resp.json().get("detail", "")
        assert "not found" in err_msg.lower()


def test_xss_payload_returned_as_inert_data(client: TestClient):
    """Verify that notes with raw script tags return literal text data."""
    # Search for XSS note
    resp_search = client.get("/api/search?q=inert+payload")
    assert resp_search.status_code == 200
    results = resp_search.json()["results"]
    assert len(results) > 0
    xss_res = results[0]
    # The snippet contains raw literal characters, never interpreted as HTML
    assert "<script>alert(1)</script>" in xss_res["snippet"]

    # Fetch note
    resp_note = client.get("/api/note/xss-security-test")
    assert resp_note.status_code == 200
    content = resp_note.json()["content"]
    assert "<script>alert(1)</script>" in content
    assert "<img src=x onerror=alert(2)>" in content


def test_graph_endpoint(client: TestClient):
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "orphan_notes" in data
    assert "broken_links" in data
    assert len(data["nodes"]) >= 18
    assert "isolated-orphan-concept.md" in data["orphan_notes"]


def test_api_and_cli_search_parity(client: TestClient, api_test_dir: Path):
    """Assert that core SearchEngine (used by CLI) and API return identical scores and ordering."""
    from brain.search import SearchEngine
    from brain.index import Index

    idx = Index(api_test_dir)
    idx.load()
    engine = SearchEngine(idx)

    query = "machine learning"
    engine_results = engine.search(query, k=5)
    api_resp = client.get(f"/api/search?q={query}&k=5")
    assert api_resp.status_code == 200
    api_results = api_resp.json()["results"]

    assert len(engine_results) == len(api_results)
    for eng_r, api_r in zip(engine_results, api_results):
        assert eng_r.path == api_r["path"]
        assert pytest.approx(round(eng_r.score, 4), rel=1e-5) == api_r["score"]


def test_rate_limiter_returns_429(client: TestClient):
    """Verify that requests exceeding 60 per minute per IP receive HTTP 429."""
    headers = {"X-Forwarded-For": "203.0.113.99"}

    # First 60 requests should pass through
    for i in range(60):
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 200, f"Request {i+1} failed with {resp.status_code}"

    # 61st request from same IP must receive 429 Rate Limit Exceeded
    r_limited = client.get("/health", headers=headers)
    assert r_limited.status_code == 429
    data = r_limited.json()
    assert "error" in data
    assert "rate limit exceeded" in data["error"].lower()
    assert "Retry-After" in r_limited.headers


def test_csp_hash_matches_served_html_bytes(client: TestClient):
    """Verify that the inline-script SHA-256 hash in CSP matches the exact bytes of the served HTML."""
    import base64
    import hashlib

    resp = client.get("/")
    assert resp.status_code == 200
    html_bytes = resp.content

    # Extract exact bytes between <script> and </script>
    start_tag = b"<script>"
    end_tag = b"</script>"
    start_pos = html_bytes.find(start_tag)
    end_pos = html_bytes.find(end_tag, start_pos)
    assert start_pos != -1, "No <script> tag found in served HTML"
    assert end_pos != -1, "No </script> tag found in served HTML"

    script_bytes = html_bytes[start_pos + len(start_tag):end_pos]
    # Browsers hash the script after normalizing line endings to LF, so the
    # CSP token must match the normalized bytes (not raw CRLF bytes on Windows).
    script_bytes = script_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest = hashlib.sha256(script_bytes).digest()
    b64_digest = base64.b64encode(digest).decode("ascii")
    expected_token = f"'sha256-{b64_digest}'"

    csp = resp.headers.get("content-security-policy", "")
    assert expected_token in csp, f"Expected CSP token {expected_token} not found in header: {csp}"

    # Verify script-src does not contain 'unsafe-inline'
    script_src_part = [p for p in csp.split(";") if "script-src" in p]
    assert script_src_part, "No script-src directive in CSP header"
    assert "'unsafe-inline'" not in script_src_part[0], f"'unsafe-inline' unexpectedly found in script-src: {script_src_part[0]}"


def test_rate_limiter_spoofed_forwarded_for(client: TestClient):
    """Verify rate limiter is resistant to spoofed X-Forwarded-For headers (extracts rightmost IP)."""
    # Attacker attempts to rotate fake leftmost IPs: "10.0.0.1", "10.0.0.2", etc.
    # Cloud Run / reverse proxy appends real connecting IP "198.51.100.77" to the end
    for i in range(60):
        spoofed_header = {"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.77"}
        resp = client.get("/health", headers=spoofed_header)
        assert resp.status_code == 200, f"Request {i+1} failed"

    # 61st request with yet another fake prefix must still be blocked under real IP 198.51.100.77
    blocked = client.get("/health", headers={"X-Forwarded-For": "10.0.0.999, 198.51.100.77"})
    assert blocked.status_code == 429
    assert "rate limit exceeded" in blocked.json()["error"].lower()


def test_rate_limiter_memory_eviction():
    """Verify that expired IP records are purged so memory cannot grow without bound."""
    from brain.api import RateLimitMiddleware
    from collections import deque
    import time

    middleware = RateLimitMiddleware(app=None, max_requests=10, window_seconds=60, max_tracked_ips=5)
    now = time.time()

    # Add expired IPs (active 120 seconds ago)
    middleware.ip_timestamps["192.0.2.1"] = deque([now - 120, now - 110])
    middleware.ip_timestamps["192.0.2.2"] = deque([now - 90])

    # Add an active IP (active 10 seconds ago)
    middleware.ip_timestamps["192.0.2.3"] = deque([now - 10])

    assert len(middleware.ip_timestamps) == 3

    # Trigger cleanup
    middleware._cleanup_stale_entries(now)

    # Expired IPs must be evicted; active IP must remain
    assert "192.0.2.1" not in middleware.ip_timestamps
    assert "192.0.2.2" not in middleware.ip_timestamps
    assert "192.0.2.3" in middleware.ip_timestamps
    assert len(middleware.ip_timestamps) == 1

    # Verify hard ceiling enforcement
    for i in range(10):
        middleware.ip_timestamps[f"10.1.{i}.1"] = deque([now])
    assert len(middleware.ip_timestamps) == 11
    middleware._cleanup_stale_entries(now)
    assert len(middleware.ip_timestamps) <= middleware.max_tracked_ips




def test_note_body_strips_frontmatter(client: TestClient):
    """The viewer gets a body without the YAML block; raw content is unchanged."""
    resp = client.get("/api/note/spaced-repetition")
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"].startswith("---")
    assert "tags:" in data["content"]
    assert not data["body"].startswith("---")
    assert "tags: [" not in data["body"]
    assert data["body"].startswith("# Spaced Repetition")


def test_note_body_without_frontmatter_is_unchanged(client: TestClient, api_test_dir: Path):
    resp = client.get("/api/note/isolated-orphan-concept")
    assert resp.status_code == 200
    data = resp.json()
    raw = (api_test_dir / "isolated-orphan-concept.md").read_text(encoding="utf-8")
    if not raw.lstrip().startswith("---"):
        assert data["body"] == raw.lstrip("\r\n")


def test_csp_hash_ignores_line_endings(tmp_path: Path):
    """CRLF (Windows checkout) and LF versions of the page must give the same CSP hash."""
    from brain.api import compute_inline_script_sha256

    script = "\n  const x = 1;\n  console.log(x);\n"
    lf = tmp_path / "lf.html"
    crlf = tmp_path / "crlf.html"
    lf.write_bytes(f"<html><script>{script}</script></html>".encode())
    crlf.write_bytes(f"<html><script>{script}</script></html>".replace("\n", "\r\n").encode())
    assert compute_inline_script_sha256(lf) == compute_inline_script_sha256(crlf) != ""
