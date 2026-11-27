# Contributing to Brain

Thank you for collaborating on **Brain**, our local Markdown Second Brain CLI, TF-IDF search engine, and knowledge graph visualizer!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amarpai850/markdown-secondbrain-CLI.git
   cd markdown-secondbrain-CLI
   ```

2. **Create and activate a virtual environment (Python 3.11+)**
   ```bash
   python -m venv .venv
   ```
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install in editable mode with all optional extras**
   ```bash
   pip install -e ".[web,semantic,dev]"
   ```

4. **Verify the test suite**
   ```bash
   pytest -v
   ```

## Workflow & Collaboration Rules

- **Branch Naming:** Always create a new branch from `main` using the convention `feature/<short-description>` (e.g., `feature/tag-boosting`, `feature/ui-frontmatter-strip`).
- **Small, Focused PRs:** Keep pull requests small and scoped to a single feature, bugfix, or doc update so they are easy to review.
- **Peer Code Review:** Both collaborators review each other's PRs before merging into `main`.
- **Pre-Push Check:** Always run the full `pytest` suite locally before every push:
  ```bash
  pytest -v
  ```
  All tests must pass with zero failures before pushing.
- **Keep Personal Data Out:** Never commit personal notes, `.env` files, `.brain/` index directories, or local agent scratch files. Only `sample_notes/` belongs in the repository.
