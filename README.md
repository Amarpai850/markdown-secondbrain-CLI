# 🧠 Brain: Local Markdown "Second Brain" CLI & Search Engine

> A high-performance, offline-first personal knowledge management (PKM) CLI, search engine, and interactive graph visualizer built in Python with a zero-dependency web interface.

---

## Architecture Overview

```
                           +----------------------------------------+
                           |           User Interfaces              |
                           |   CLI (`brain`)  |  Web UI (HTML5/JS)   |
                           +---------------------+------------------+
                                                 |
                                       +---------v---------+
                                       |  FastAPI (api.py) |
                                       +---------+---------+
                                                 |
+------------------------------------------------v----------------------------------------------+
|                                    Core Engine (Standard Library)                             |
|                                                                                               |
|  +--------------------+    +--------------------+    +-------------------+    +------------+  |
|  | Scanner            |--->| Parser             |--->| Tokenizer &       |--->| Inverted   |  |
|  | (mtime, walk)      |    | (links, tags, meta)|    | Stemmer (scratch) |    | Index      |  |
|  +--------------------+    +--------------------+    +-------------------+    +-----+------+  |
|                                                                                     |         |
|  +--------------------+    +--------------------+                                   |         |
|  | ANSI Highlight     |    | Graph Engine       |<----------------------------------+         |
|  | (terminal snippet) |    | (backlinks, BFS)   |                                   |         |
|  +--------------------+    +--------------------+                                   v         |
|                                                                                +------------+ |
|                                                                                | Search &   | |
|                                                                                | TF-IDF     | |
|                                                                                | Ranking    | |
|                                                                                +------------+ |
+-----------------------------------------------------------------------------------------------+
                                                 |
                                                 v
                               +-----------------------------------+
                               | Persistence: `<dir>/.brain/index.json`
                               +-----------------------------------+
```

---

## Features

- **Pure Standard-Library Core**: Zero search or NLP dependencies for core operation (no `scikit-learn`, `whoosh`, or `nltk`).
- **TF-IDF Vector Space Model from Scratch**: Sublinear term frequencies, smoothed inverse document frequency, and $3\times$ title and heading boosting.
- **Incremental Indexing**: Tracks file `mtime` and `size` to update only changed notes, while recomputing global weights and norms.
- **Knowledge Graph Analytics**: Bidirectional `[[wiki-links]]` resolution, backlinks, orphan note detection, broken link detection, and BFS neighborhood discovery.
- **Optional Semantic Search (LSA)**: Latent Semantic Analysis via Truncated SVD with query folding ($q_k = \Sigma^{-1} U^T q$) using `numpy` as an optional dependency.
- **Interactive Terminal UI**: ANSI keyword highlighting, Windows Virtual Terminal Processing (VTP) support, and automatic `NO_COLOR` handling.
- **Thin Web Layer**: FastAPI backend with zero-CDN single-file vanilla HTML5/CSS/JS frontend featuring a force-directed canvas graph with cooling physics.

---

## Installation

### Prerequisites
- Python 3.11+
- Git

### Local Installation
```bash
# Clone the repository
git clone <repo-url>
cd "SEARCH ENGINE"

# Create a local virtual environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Install core CLI in editable mode
pip install -e .

# Install with web, semantic, and development extras
pip install -e ".[web,semantic,dev]"
```

---

## CLI Usage Guide

The package provides a `brain` terminal command:

### 1. Indexing Notes
Scan a directory of Markdown files and construct/update the inverted index:
```bash
# Index sample notes (or specify any markdown folder)
brain index sample_notes

# Force a complete index rebuild from scratch
brain index sample_notes --force
```

### 2. Ranked Search
Search notes using cosine similarity over TF-IDF vectors:
```bash
# Basic keyword search
brain search "machine learning" --dir sample_notes

# Limit top-k results and filter by tag
brain search "deep learning" -k 5 --tag nlp --dir sample_notes

# Optional semantic LSA search (requires numpy)
brain search "neural representations" --semantic --dir sample_notes
```

### 3. Link Inspection
Inspect outgoing wiki-links and incoming backlinks for a given note:
```bash
brain links transformers --dir sample_notes
```

### 4. Knowledge Graph Statistics
View aggregate graph topology, orphan notes, broken links, and top-connected notes:
```bash
brain graph --stats --dir sample_notes
```

### 5. Breadth-First Neighbor Discovery
Explore connected notes up to $N$ hops away using BFS:
```bash
# Traversal following both incoming and outgoing links (default)
brain neighbors transformers --depth 2 --dir sample_notes

# Traversal following outgoing links only
brain neighbors transformers --depth 2 --direction out --dir sample_notes

# Traversal following incoming backlinks only
brain neighbors transformers --depth 2 --direction in --dir sample_notes
```

### 6. Starting the Web UI Locally
Launch the FastAPI web application serving the single-page interface:
```bash
brain serve --host 127.0.0.1 --port 8000 --dir sample_notes
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## TF-IDF Design & Theory

### 1. Mathematical Formulation
- **Sublinear Term Frequency ($tf$)**:
  $$\text{tf}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$
- **Smoothed Inverse Document Frequency ($idf$)**:
  $$\text{idf}(t, D) = \ln\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$
  where $N$ is total indexed documents and $\text{df}(t)$ is document frequency.
- **$3\times$ Title and Heading Boost**:
  Tokens occurring in headings or the document title receive a $3\times$ multiplier in effective count before sublinear log scaling. Headings are stripped from body text beforehand to prevent double counting:
  $$\text{effective\_count}(t, d) = \text{count}_{\text{body}}(t, d) + 3 \times \text{count}_{\text{headings}}(t, d) + 3 \times \text{count}_{\text{title}}(t, d)$$
- **$L_2$ Vector Normalization**:
  $$w(t, d) = \text{tf}(t, d) \times \text{idf}(t, D)$$
  $$\mathbf{v}_d = \frac{[w(t_1, d), w(t_2, d), \dots, w(t_m, d)]}{\sqrt{\sum_{k} w(t_k, d)^2}}$$
- **Cosine Similarity Ranking**:
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{v}_d = \sum_{t \in q \cap d} q_t \cdot v_{d, t}$$

### 2. Incremental Re-Indexing Algorithm
- File modification times (`mtime`) and file sizes are stored in `<dir>/.brain/index.json`.
- On re-index, `brain` identifies `added`, `modified`, and `deleted` files without re-reading unchanged files.
- If changes occur, term counts are updated, and global $df$, $idf$, document weights, and norms are recomputed. Strict mathematical parity against a full rebuild is verified in unit tests.

### 3. Latent Semantic Analysis (LSA) & Query Folding
- Term-document matrix $A \in \mathbb{R}^{M \times N}$ is decomposed via Truncated SVD: $A \approx U_k \Sigma_k V_k^T$.
- Query vectors are folded into latent semantic space via:
  $$\mathbf{q}_k = \Sigma_k^{-1} U_k^T \mathbf{q}$$
- Cosine similarity is computed between $\hat{\mathbf{q}}_k$ and latent document vectors $\hat{\mathbf{d}}_j$.
- SVD matrices are cached in memory and automatically invalidated whenever index changes occur.

---

## Security Architecture

1. **Path Traversal Prevention**:
   - The `/api/note/{name}` endpoint resolves notes *strictly* through the in-memory index dictionary (`graph.resolve_note(name)`).
   - Arbitrary filesystem path joins (`os.path.join(base_dir, name)`) are never executed. Traversal attempts (e.g. `../../etc/passwd` or `..%2f..%2fpyproject.toml`) fail resolution and return an immediate JSON 404 without disk interaction.
2. **XSS Protection**:
   - The single-page frontend builds search snippets using safe DOM manipulation: `document.createTextNode()` and `<mark>` elements populated via `textContent`.
   - `innerHTML` is never used with note content, rendering injected script payloads (e.g. `<script>alert(1)</script>`) completely inert.
3. **Content-Security-Policy (CSP)**:
   - Evaluates a cryptographic SHA-256 hash of the inline script in `web/index.html` at startup:
     ```http
     Content-Security-Policy: default-src 'self'; script-src 'self' 'sha256-...'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'
     X-Content-Type-Options: nosniff
     ```
4. **Abuse Protection & Rate Limiting**:
   - In-memory sliding window rate limiter protects endpoints: caps client IPs at **60 requests per minute**. Exceeded limits return HTTP 429 with `Retry-After: 60`.
   - Query lengths are validated strictly to 1–200 characters; $k$ is clamped to 1–50.
5. **Deployment Isolation**:
   - The cloud container copies *only* `sample_notes/` into `/app/sample_notes`. Personal notes and files are ignored via `.dockerignore`.
   - The container runs as an unprivileged non-root user (`appuser`, UID 1000) on a read-only filesystem at request time.

---

## Cloud Deployment Guide

### Option 1: Google Cloud Run (Recommended)

Cloud Run compiles and deploys the container remotely via Cloud Build using `gcloud run deploy --source .`.

#### Step-by-Step Manual Setup in PowerShell:
```powershell
# 1. Install Google Cloud SDK if not present
winget install Google.CloudSDK

# 2. Authenticate with Google Cloud
gcloud auth login

# 3. Create or select a Google Cloud Project
gcloud projects create my-brain-demo-2026 --name="Brain Search"
gcloud config set project my-brain-demo-2026

# 4. Link Billing Account:
# Open https://console.cloud.google.com/billing/linkedaccount?project=my-brain-demo-2026

# 5. Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 6. Set default deployment region
gcloud config set run/region asia-south1

# 7. Deploy directly from current directory
gcloud run deploy brain `
  --source . `
  --allow-unauthenticated `
  --max-instances=2 `
  --memory=512Mi `
  --cpu=1 `
  --timeout=30 `
  --port=8080
```

#### Flags Explained & Cost Implications:
- `--allow-unauthenticated`: Permits public HTTPS access to the web demo without requiring Google IAM authentication.
- `--max-instances=2`: Restricts horizontal auto-scaling to at most 2 container instances, eliminating the risk of runaway traffic bills.
- `--memory=512Mi`: Allocates 512 MB RAM per instance, optimal for the in-memory index.
- `--cpu=1`: Allocates 1 vCPU per container instance.
- `--timeout=30`: Drops hanging connections after 30 seconds.
- `--port=8080`: Routes incoming HTTP traffic to container port 8080.
- **Cost**: Cloud Run includes an **Always Free Tier** each month (2 million requests, 360,000 GB-seconds of memory, 180,000 vCPU-seconds). Because the service scales to 0 instances when idle, this demo stays $0/month.

#### Setting Budget Alerts & Teardown:
- **Budget Alert**: In Google Cloud Console > **Billing** > **Budgets & Alerts** > Create budget (e.g. $1.00 limit with email notifications at 50%, 90%, 100%).
- **Service Teardown**: To delete the Cloud Run service:
  ```powershell
  gcloud run services delete brain --region asia-south1 --quiet
  ```
- **Important: Artifact Registry & Cloud Storage Leftovers**:
  When deploying with `gcloud run deploy --source .`, Google Cloud Build automatically stores your built container images in **Artifact Registry** (repository: `cloud-run-source-deploy`) and uploads source tarballs and build logs to **Cloud Storage** buckets (`<PROJECT_ID>_cloudbuild` and `artifacts.<PROJECT_ID>.appspot.com`).
  Deleting only the Cloud Run service leaves these container images and build logs behind, which can incur small ongoing storage charges.
- **Complete Project Teardown (Zero Leftovers)**:
  To eliminate all resources, storage buckets, container repositories, and any chance of lingering costs in one step:
  ```powershell
  gcloud projects delete $PROJECT_ID
  ```

---

### Option 2: Render (Free Tier Alternative)

A complete `render.yaml` specification is included for deployment on Render:

1. Push your repository to GitHub:
   ```bash
   git remote add origin https://github.com/<your-username>/brain.git
   git push -u origin master
   ```
2. Log into [dashboard.render.com](https://dashboard.render.com/).
3. Click **New +** > **Web Service** > Connect your GitHub repository.
4. Select **Docker** environment and **Free** instance type.
5. Set Environment Variables:
   - `PORT`: `10000`
   - `BRAIN_NOTES_DIR`: `/app/sample_notes`
   - `BRAIN_WEB_DIR`: `/app/web`
6. Click **Deploy Web Service**. Render builds the multi-stage Dockerfile and provides a public `https://<service-name>.onrender.com` URL.

---

## Work Split

Remaining work is divided equally between both collaborators:

| Person A (Amarpai850) | Person B (Partner) |
| :--- | :--- |
| Docker and Google Cloud Run deployment (`gcloud run deploy --source .`) | LSA / semantic search tuning (rank/k selection & edge cases) |
| Live deployed endpoint verification (`/health`, `/api/search`, `/api/graph`, path traversal checks) | Tag boosting for search ranking |
| Rate-limit checks against the live reverse-proxy environment | Web UI polish (strip YAML frontmatter in the note viewer) |
| Final deployment walkthrough & teardown verification | Extra sample notes, unit/integration tests, and README improvements |

---

## Test Suite Summary

The automated test suite contains **62 tests** across 9 test modules:

```text
============================= 62 passed in 9.48s =============================
- tests/test_tokenizer.py: 8 unit tests (stopwords, plurals, short words, participles, derivational suffixes, tightened -er protection)
- tests/test_parser.py: 6 unit tests (YAML frontmatter, code stripping, heading removal, title hierarchy, wiki-links, tags)
- tests/test_scanner.py: 2 unit tests (directory traversal, manifest change diffing)
- tests/test_tfidf.py: 4 unit tests (hand-verified mathematical checks for DF, IDF, TF, L2 norms, cosine scores, and 3x boost)
- tests/test_search.py: 6 unit tests (ranking, tag filtering, snippet density, LSA SVD, cache invalidation, numpy-free keyword search)
- tests/test_graph.py: 7 unit tests (nodes/edges, orphan detection, broken link detection, backlinks, BFS traversal, web export)
- tests/test_index_incremental.py: 1 integration test (parity between incremental updates and fresh rebuilds)
- tests/test_cli.py: 11 integration tests (subprocess execution of all subcommands and error handling)
- tests/test_api.py: 17 integration tests (FastAPI endpoints, input validation, path traversal attempts, XSS inertness, CSP SHA-256 verification, spoof-resistant rate limiting, memory eviction, and CLI/API score parity)
```
