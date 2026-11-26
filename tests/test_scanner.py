"""Unit tests for file scanner and incremental change manifest detection."""

from pathlib import Path
from brain.scanner import FileMeta, diff_manifests, scan_directory


def test_scan_sample_notes():
    sample_dir = Path("sample_notes").resolve()
    manifest = scan_directory(sample_dir)
    # Should discover all 18 sample notes
    assert len(manifest) == 18
    assert "attention-mechanisms.md" in manifest
    assert "isolated-orphan-concept.md" in manifest

    meta = manifest["attention-mechanisms.md"]
    assert meta.rel_path == "attention-mechanisms.md"
    assert meta.size > 0
    assert meta.mtime > 0


def test_diff_manifests():
    old = {
        "note1.md": FileMeta("note1.md", 100.0, 500),
        "note2.md": FileMeta("note2.md", 200.0, 600),
        "deleted.md": FileMeta("deleted.md", 300.0, 400),
    }
    new = {
        "note1.md": FileMeta("note1.md", 100.0, 500),  # Unchanged
        "note2.md": FileMeta("note2.md", 250.0, 650),  # Modified
        "added.md": FileMeta("added.md", 400.0, 300),   # Added
    }

    added, modified, deleted = diff_manifests(old, new)
    assert added == {"added.md"}
    assert modified == {"note2.md"}
    assert deleted == {"deleted.md"}
