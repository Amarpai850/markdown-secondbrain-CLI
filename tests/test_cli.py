"""Integration tests for CLI executed via subprocess."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def cli_temp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary directory with a space in its path and a fresh copy of sample_notes."""
    base_tmp = tmp_path_factory.mktemp("brain cli tests")
    notes_copy = base_tmp / "my sample notes"
    sample_src = Path("sample_notes").resolve()
    shutil.copytree(sample_src, notes_copy)
    return notes_copy


def run_brain_cli(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run CLI command via sys.executable -m brain."""
    cmd = [sys.executable, "-m", "brain", *args]
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=run_env,
    )


def test_cli_help():
    proc = run_brain_cli(["--help"])
    assert proc.returncode == 0
    assert "usage: brain" in proc.stdout.lower()
    assert "index" in proc.stdout
    assert "search" in proc.stdout
    assert "links" in proc.stdout
    assert "graph" in proc.stdout
    assert "neighbors" in proc.stdout
    assert "serve" in proc.stdout


def test_cli_index_command(cli_temp_dir: Path):
    proc = run_brain_cli(["index", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Indexed 18 documents" in proc.stdout
    # Verify .brain/index.json created in temp dir, NOT real sample_notes
    assert (cli_temp_dir / ".brain" / "index.json").exists()


def test_cli_search_command(cli_temp_dir: Path):
    proc = run_brain_cli(["search", "attention mechanism", "--dir", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Search results for 'attention mechanism'" in proc.stdout
    assert "attention-mechanisms.md" in proc.stdout or "transformers.md" in proc.stdout


def test_cli_search_with_tag_and_k(cli_temp_dir: Path):
    proc = run_brain_cli(["search", "learning", "-k", "3", "--tag", "nlp", "--dir", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Search results for 'learning'" in proc.stdout
    assert "[tag: #nlp]" in proc.stdout


def test_cli_links_command(cli_temp_dir: Path):
    proc = run_brain_cli(["links", "transformers", "--dir", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Links for The Transformer Architecture" in proc.stdout or "Links for" in proc.stdout
    assert "Outgoing Links" in proc.stdout
    assert "Backlinks" in proc.stdout
    assert "attention-mechanisms.md" in proc.stdout


def test_cli_graph_stats_command(cli_temp_dir: Path):
    proc = run_brain_cli(["graph", "--stats", "--dir", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Knowledge Graph Statistics" in proc.stdout
    assert "Total Notes (Nodes):      18" in proc.stdout
    assert "isolated-orphan-concept.md" in proc.stdout
    assert "non-existent-architecture" in proc.stdout


def test_cli_neighbors_command(cli_temp_dir: Path):
    proc = run_brain_cli(["neighbors", "transformers", "--depth", "2", "--dir", str(cli_temp_dir)])
    assert proc.returncode == 0
    assert "Neighbors of" in proc.stdout
    assert "Hop 1" in proc.stdout
    assert "attention-mechanisms.md" in proc.stdout


def test_cli_error_missing_index(tmp_path: Path):
    empty_dir = tmp_path / "empty notes"
    empty_dir.mkdir()
    proc = run_brain_cli(["search", "query", "--dir", str(empty_dir)])
    assert proc.returncode != 0
    assert "No index found at" in proc.stderr
    # Must not output traceback
    assert "Traceback" not in proc.stderr


def test_cli_error_unknown_note(cli_temp_dir: Path):
    proc = run_brain_cli(["links", "completely-fictional-note-xyz", "--dir", str(cli_temp_dir)])
    assert proc.returncode != 0
    assert "Note 'completely-fictional-note-xyz' not found in index." in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_error_empty_query(cli_temp_dir: Path):
    proc = run_brain_cli(["search", "   ", "--dir", str(cli_temp_dir)])
    assert proc.returncode != 0
    assert "Search query cannot be empty." in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_error_bad_directory():
    proc = run_brain_cli(["index", "Z:/non/existent/path/xyz_brain_test"])
    assert proc.returncode != 0
    assert "does not exist." in proc.stderr
    assert "Traceback" not in proc.stderr
