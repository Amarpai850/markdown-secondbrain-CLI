"""Markdown parser for frontmatter, titles, headings, body, wiki-links, and tags."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedNote:
    """Structured representation of a parsed markdown note."""

    path: str  # Relative POSIX path
    title: str  # Document title
    headings: list[str] = field(default_factory=list)  # Heading texts
    body: str = ""  # Body text with frontmatter AND headings stripped
    raw_content: str = ""  # Full original document content
    links: list[str] = field(default_factory=list)  # Unique target wiki-link names
    tags: list[str] = field(default_factory=list)  # Unique lowercase tags without '#'
    frontmatter: dict[str, Any] = field(default_factory=dict)


def _strip_code(text: str) -> str:
    """Strip fenced code blocks and inline code spans to avoid false tag/link matches."""
    # Strip fenced code blocks ``` ... ```
    cleaned = re.sub(r"```[\s\S]*?```", " ", text)
    # Strip inline code ` ... `
    cleaned = re.sub(r"`[^`\n]+`", " ", cleaned)
    return cleaned


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract frontmatter and remaining text from markdown content.

    Supports YAML-like 'key: value' pairs and list formats without external dependencies.
    """
    frontmatter: dict[str, Any] = {}
    content_stripped = content.lstrip()

    if not content_stripped.startswith("---"):
        return frontmatter, content

    # Find closing ---
    match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?", content_stripped)
    if not match:
        return frontmatter, content

    fm_raw = match.group(1)
    body = content_stripped[match.end():]

    current_list_key: str | None = None
    for line in fm_raw.splitlines():
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue

        # Handle list item under an active key: "  - value"
        if line_str.startswith("- ") and current_list_key:
            item = line_str[2:].strip().strip("\"'")
            if isinstance(frontmatter[current_list_key], list):
                frontmatter[current_list_key].append(item)
            continue

        if ":" in line_str:
            key, val = line_str.split(":", 1)
            key = key.strip()
            val = val.strip()

            # Inline list: tags: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
                frontmatter[key] = items
                current_list_key = None
            elif not val:
                # Potential start of indented list
                frontmatter[key] = []
                current_list_key = key
            else:
                clean_val = val.strip("\"'")
                frontmatter[key] = clean_val
                current_list_key = None

    return frontmatter, body


def extract_wiki_links(text: str) -> list[str]:
    """Extract and canonicalize target names from [[wiki-links]].

    Strips code blocks/inline code first. Handles [[Target|Alias]] and [[Target#Section]].
    """
    clean_text = _strip_code(text)
    raw_matches = re.findall(r"\[\[(.*?)\]\]", clean_text)
    links: list[str] = []
    seen: set[str] = set()

    for item in raw_matches:
        link_str = item.strip()
        if not link_str:
            continue
        # Strip alias [[Target|Alias]]
        if "|" in link_str:
            link_str = link_str.split("|", 1)[0].strip()
        # Strip anchor [[Target#Section]]
        if "#" in link_str:
            link_str = link_str.split("#", 1)[0].strip()
        # Strip optional .md extension
        if link_str.lower().endswith(".md"):
            link_str = link_str[:-3].strip()

        if link_str and link_str.lower() not in seen:
            seen.add(link_str.lower())
            links.append(link_str)

    return links


def extract_tags(text: str, frontmatter_tags: Any = None) -> list[str]:
    """Extract lowercase tags from frontmatter and inline #tag occurrences.

    Strips code blocks/inline code before extracting inline tags.
    Strictly ignores markdown headers (# Heading).
    """
    tags_set: set[str] = set()

    # Process frontmatter tags
    if frontmatter_tags:
        if isinstance(frontmatter_tags, list):
            for t in frontmatter_tags:
                tag_cleaned = str(t).strip().lstrip("#").lower()
                if tag_cleaned:
                    tags_set.add(tag_cleaned)
        elif isinstance(frontmatter_tags, str):
            for t in frontmatter_tags.split(","):
                tag_cleaned = t.strip().lstrip("#").lower()
                if tag_cleaned:
                    tags_set.add(tag_cleaned)

    # Process inline tags after stripping code
    clean_text = _strip_code(text)
    # Match #tag at start of line or preceded by whitespace, followed by word boundary
    inline_matches = re.findall(r"(?:^|\s)#([a-zA-Z0-9_\-]+)(?=\s|$|[.,;:!?\)])", clean_text)
    for tag in inline_matches:
        # Ignore purely numeric tags (e.g. #1)
        if not tag.isdigit():
            tags_set.add(tag.lower())

    return sorted(tags_set)


def parse_markdown(content: str, rel_path: str = "") -> ParsedNote:
    """Parse a markdown document into a structured ParsedNote."""
    frontmatter, remaining_text = parse_frontmatter(content)

    # Extract headings (level 1-6)
    headings = re.findall(r"^#{1,6}\s+(.+)$", remaining_text, flags=re.MULTILINE)
    headings = [h.strip() for h in headings if h.strip()]

    # Title resolution:
    # 1. Frontmatter title
    # 2. First H1 heading
    # 3. Filename stem
    title: str = ""
    if "title" in frontmatter and str(frontmatter["title"]).strip():
        title = str(frontmatter["title"]).strip()
    else:
        h1_match = re.search(r"^#\s+(.+)$", remaining_text, flags=re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()
        elif rel_path:
            title = Path(rel_path).stem
        else:
            title = "Untitled"

    # Remove headings from body text so they aren't double-counted during TF-IDF weighting
    body_no_headings = re.sub(r"^#{1,6}\s+.*$", " ", remaining_text, flags=re.MULTILINE)

    # Extract links and tags
    links = extract_wiki_links(remaining_text)
    tags = extract_tags(remaining_text, frontmatter.get("tags"))

    return ParsedNote(
        path=rel_path,
        title=title,
        headings=headings,
        body=body_no_headings.strip(),
        raw_content=content,
        links=links,
        tags=tags,
        frontmatter=frontmatter,
    )
