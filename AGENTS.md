# AGENTS.md — AI Coding Agent Context

This file provides shared architectural context, hard constraints, and project status for AI coding assistants (Antigravity, Claude Code, etc.) working in this repository.

## Project Goal
**Brain** is a local-first Markdown "Second Brain" CLI, from-scratch TF-IDF search engine, and directed `[[wiki-link]]` knowledge graph, wrapped in a thin FastAPI + vanilla HTML/CSS/JS web layer for optional cloud deployment. The core search and graph engine must work 100% offline with zero external search libraries.

## Module Map (`src/brain/`)
- `__init__.py` — Package version and public exports.
- `__main__.py` — Entrypoint for `python -m brain`.
- `scanner.py` — Recursive `.md` directory traversal (`pathlib`) and `mtime`/size manifest diffing for incremental indexing.
- `parser.py` — Strips YAML frontmatter, fenced code blocks, and inline code before extracting `#tags` and `[[wiki-links]]`; strips headings from body text to prevent double counting the 3x heading boost.
- `tokenizer.py` — Lowercasing, regex tokenization, built-in stopwords list, and custom rule-based suffix stemmer (including tightened `-er` rules).
- `index.py` — Inverted index storing raw boosted term counts per document; recomputes DF, IDF, TF-IDF weights, and L2 norms across all documents after any add/modify/delete; atomic JSON persistence to `.brain/index.json`.
- `search.py` — Cosine similarity ranking over L2-normalized TF-IDF vectors, tag filtering, densest ~150-char snippet extraction with match offsets, and optional LSA semantic search (`q_k = Sigma^-1 U^T q` with SVD cache invalidated on index change).
- `graph.py` — Directed knowledge graph tracking outlinks, backlinks, orphans, broken links, degree centrality, and BFS `neighbors(note, depth, direction="both"|"out"|"in")`.
- `highlight.py` — Terminal ANSI keyword highlighting with Windows Virtual Terminal Processing, `NO_COLOR` support, and automatic TTY detection.
- `cli.py` — `argparse` CLI (`index`, `search`, `links`, `graph --stats`, `orphans`, `neighbors --direction`, `serve`) with UTF-8 console output and user-friendly errors.
- `api.py` — FastAPI app loading the index read-only at startup via lifespan; serves `/`, `/health`, `/api/search`, `/api/note/{name}`, `/api/graph`; enforces path traversal protection (index-only lookup), SHA-256 inline-script CSP, and sliding-window rate limiting.

## Hard Rules
1. **Standard Library Core:** Core modules (`scanner`, `parser`, `tokenizer`, `index`, `search`, `graph`, `highlight`, `cli`) must use **only** the Python 3.11+ standard library.
2. **From-Scratch Algorithms:** Tokenization, stemming, TF-IDF weighting, cosine ranking, and snippet extraction must remain implemented from scratch. Do **not** add `scikit-learn`, `whoosh`, `nltk`, or similar libraries.
3. **Optional Extras Only:** `fastapi`/`uvicorn` belong strictly in `[web]` and `numpy` (for optional LSA) in `[semantic]`. Keyword search and all CLI commands (except `serve`) must work without them.
4. **Test Integrity:** Never modify or weaken existing tests to make new code pass without an explicit, stated reason.
5. **Deployment & Privacy:** Only `sample_notes/` may be bundled in Docker/cloud deployments. Never commit or deploy personal notes.
6. **Security Invariants:**
   - `/api/note/{name}` must resolve strictly through the in-memory index, never via filesystem path joins.
   - `web/index.html` must render note text and snippets via DOM nodes (`textContent` + `<mark>`), never `innerHTML`.

## How to Run Tests
```bash
# Run full test suite (must pass before every push)
pytest -v

# Run a specific test module
pytest tests/test_api.py -v
```

## Current Status
### Done
- **Phase 1–3 (Core & CLI):** Scanner, parser, from-scratch tokenizer/stemmer, inverted index with incremental updates, TF-IDF + optional LSA search, directed knowledge graph, ANSI highlighter, and full CLI (`index`, `search`, `links`, `graph`, `neighbors`, `serve`).
- **Phase 4 (Web & Security):** FastAPI read-only wrapper + single-file dark UI (`web/index.html`) with force-directed `<canvas>` graph, SHA-256 CSP header, XSS-safe DOM rendering, path-traversal protection, and X-Forwarded-For-hardened rate limiter with bounded memory.
- **Phase 5 (Packaging & Docs):** Multi-stage `Dockerfile` (Python 3.11 test gate + non-root runtime), `.dockerignore`, `render.yaml`, `README.md`, `CONTRIBUTING.md`, and `AGENTS.md`.
- **Tests:** 75 tests passing across 9 test modules (`pytest -v`).

### Remaining
- ~~**Deployment & Live Verification:**~~ Done: live on Render at https://brain-search-engine.onrender.com; `/health`, search (keyword + semantic), notes, graph, CSP headers, traversal 404s, query validation and rate limiting verified live.
- ~~**Feature & Polish Enhancements (Person B):**~~ Done: LSA rank selection and edge cases, tag boosting, frontmatter-free note viewer (`body` field on `/api/note`), two extra sample notes (20 total), and tests.
