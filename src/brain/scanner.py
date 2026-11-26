"""Directory scanner for markdown files with mtime tracking and incremental change detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Standard directories to ignore when scanning notes
IGNORE_DIRS: frozenset[str] = frozenset({
    ".brain",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
})


@dataclass
class FileMeta:
    """Metadata for an indexed file tracking relative path, modification time, and size."""

    rel_path: str  # POSIX relative path
    mtime: float  # Last modification timestamp
    size: int  # File size in bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileMeta:
        """Deserialize from dictionary."""
        return cls(
            rel_path=data["rel_path"],
            mtime=float(data["mtime"]),
            size=int(data["size"]),
        )


def scan_directory(base_dir: Path) -> dict[str, FileMeta]:
    """Recursively scan base_dir for .md files, ignoring internal and hidden folders.

    Returns a mapping of relative POSIX path to FileMeta.
    """
    resolved_base = base_dir.resolve()
    manifest: dict[str, FileMeta] = {}

    if not resolved_base.exists() or not resolved_base.is_dir():
        return manifest

    for path in resolved_base.rglob("*.md"):
        # Check if any parent part between base and path is in IGNORE_DIRS
        try:
            rel_parts = path.relative_to(resolved_base).parts
        except ValueError:
            continue

        # Skip ignored directory trees or hidden files
        if any(part in IGNORE_DIRS or (part.startswith(".") and part != ".") for part in rel_parts[:-1]):
            continue
        if rel_parts[-1].startswith("."):
            continue

        try:
            stat = path.stat()
            rel_posix = path.relative_to(resolved_base).as_posix()
            manifest[rel_posix] = FileMeta(
                rel_path=rel_posix,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        except OSError:
            # Skip files that cannot be accessed
            continue

    return manifest


def diff_manifests(
    old_manifest: dict[str, FileMeta],
    new_manifest: dict[str, FileMeta],
) -> tuple[set[str], set[str], set[str]]:
    """Compare old and new manifests to determine changed files.

    Returns:
        (added, modified, deleted) sets of relative POSIX paths.
    """
    old_keys = set(old_manifest.keys())
    new_keys = set(new_manifest.keys())

    added = new_keys - old_keys
    deleted = old_keys - new_keys

    modified: set[str] = set()
    for key in old_keys & new_keys:
        old_item = old_manifest[key]
        new_item = new_manifest[key]
        if old_item.mtime != new_item.mtime or old_item.size != new_item.size:
            modified.add(key)

    return added, modified, deleted
