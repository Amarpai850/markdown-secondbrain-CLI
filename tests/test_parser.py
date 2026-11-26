"""Unit tests for markdown parser, frontmatter, wiki-links, tags, and headings."""

from pathlib import Path
from brain.parser import (
    extract_tags,
    extract_wiki_links,
    parse_frontmatter,
    parse_markdown,
)


def test_parse_frontmatter_yaml_variants():
    # Inline list format
    doc1 = """---
title: Machine Learning Basics
tags: [ml, ai, deep-learning]
---
# Document Body
Content here.
"""
    fm1, body1 = parse_frontmatter(doc1)
    assert fm1["title"] == "Machine Learning Basics"
    assert fm1["tags"] == ["ml", "ai", "deep-learning"]
    assert "# Document Body" in body1

    # Indented list format
    doc2 = """---
title: "Advanced Search"
tags:
  - search
  - tfidf
---
Body text here.
"""
    fm2, body2 = parse_frontmatter(doc2)
    assert fm2["title"] == "Advanced Search"
    assert fm2["tags"] == ["search", "tfidf"]
    assert "Body text here." in body2


def test_code_blocks_and_inline_code_stripped():
    """Amendment 4: Strip fenced code blocks and inline code before extracting #tags and [[links]]."""
    doc = """
Here is a normal link: [[valid-note]].
And a normal tag: #validtag.

```python
# This is a comment inside code block #fakecode
code_var = "[[fake-code-link]]"
```

Inline code here: `#fakeinlinetag` and `[[fake-inline-link]]`.
"""
    links = extract_wiki_links(doc)
    assert "valid-note" in links
    assert "fake-code-link" not in links
    assert "fake-inline-link" not in links

    tags = extract_tags(doc)
    assert "validtag" in tags
    assert "fakecode" not in tags
    assert "fakeinlinetag" not in tags


def test_headings_removed_from_body():
    """Amendment 2: Remove headings from body text before counting so 3x boost isn't double counted."""
    doc = """# Main Title

Introductory paragraph.

## Subheading Alpha

Alpha explanation.

### Deep Section

Deep explanation.
"""
    parsed = parse_markdown(doc, rel_path="test.md")
    assert parsed.title == "Main Title"
    assert parsed.headings == ["Main Title", "Subheading Alpha", "Deep Section"]
    # Verify headings are not in body text
    assert "# Main Title" not in parsed.body
    assert "Subheading Alpha" not in parsed.body
    assert "Deep Section" not in parsed.body
    # Verify body retains paragraphs
    assert "Introductory paragraph." in parsed.body
    assert "Alpha explanation." in parsed.body
    assert "Deep explanation." in parsed.body


def test_title_fallback_hierarchy():
    # 1. Frontmatter title wins
    doc1 = """---
title: Frontmatter Title
---
# H1 Title
Body
"""
    parsed1 = parse_markdown(doc1, rel_path="fallback.md")
    assert parsed1.title == "Frontmatter Title"

    # 2. H1 Title wins if no frontmatter title
    doc2 = """---
tags: [test]
---
# H1 Title Only
Body
"""
    parsed2 = parse_markdown(doc2, rel_path="fallback.md")
    assert parsed2.title == "H1 Title Only"

    # 3. Filename stem wins if neither is present
    doc3 = """No headings here, just plain text."""
    parsed3 = parse_markdown(doc3, rel_path="my-custom-note.md")
    assert parsed3.title == "my-custom-note"


def test_wiki_link_variants():
    doc = """
Links test:
- Simple: [[machine-learning]]
- With alias: [[transformers|The Transformer Architecture]]
- With section anchor: [[vector-space-models#Cosine Similarity]]
- With extension: [[graph-traversal.md]]
- Duplicate link: [[machine-learning]]
"""
    links = extract_wiki_links(doc)
    assert links == [
        "machine-learning",
        "transformers",
        "vector-space-models",
        "graph-traversal",
    ]


def test_tags_extraction_and_header_distinction():
    doc = """# Heading One is NOT a Tag

This is #machine-learning and #search-engine in action.
Ignore plain numbers like #123.
"""
    tags = extract_tags(doc, frontmatter_tags=["nlp", "AI"])
    assert tags == ["ai", "machine-learning", "nlp", "search-engine"]
