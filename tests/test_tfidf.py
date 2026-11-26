"""Mathematical verification of TF-IDF calculations against hand-computed values.

=== Hand-Worked Calculation Details ===

Corpus: 3 documents (N = 3)
Doc 1 ("doc1.md"): "cat dog"
Doc 2 ("doc2.md"): "cat cat fish"
Doc 3 ("doc3.md"): "bird fish"

Vocabulary & Document Frequencies (df):
- "cat": occurs in doc1, doc2  -> df = 2
- "dog": occurs in doc1        -> df = 1
- "fish": occurs in doc2, doc3 -> df = 2
- "bird": occurs in doc3       -> df = 1

IDF Calculation Formula: idf(t) = ln((N + 1) / (df(t) + 1)) + 1
With N = 3, N + 1 = 4:
- idf("cat"):  ln(4 / (2 + 1)) + 1 = ln(4/3) + 1 = 0.28768207245178085 + 1 = 1.2876820724517808
- idf("fish"): ln(4 / (2 + 1)) + 1 = ln(4/3) + 1 = 0.28768207245178085 + 1 = 1.2876820724517808
- idf("dog"):  ln(4 / (1 + 1)) + 1 = ln(4/2) + 1 = ln(2) + 1 = 0.6931471805599453 + 1 = 1.6931471805599454
- idf("bird"): ln(4 / (1 + 1)) + 1 = ln(4/2) + 1 = ln(2) + 1 = 0.6931471805599453 + 1 = 1.6931471805599454

TF Calculation Formula: tf(t, d) = 1 + ln(count(t, d))

Document 1 ("cat dog"):
- count("cat") = 1 -> tf = 1 + ln(1) = 1.0
  weight("cat") = 1.0 * 1.2876820724517808 = 1.2876820724517808
- count("dog") = 1 -> tf = 1 + ln(1) = 1.0
  weight("dog") = 1.0 * 1.6931471805599454 = 1.6931471805599454
- L2 norm: sqrt(1.2876820724517808^2 + 1.6931471805599454^2)
         = sqrt(1.65812512 + 2.86674737) = sqrt(4.52487249) = 2.1271747683
- L2 normalized weights:
  v1("cat") = 1.2876820724517808 / 2.1271747683 ~= 0.6053485
  v1("dog") = 1.6931471805599454 / 2.1271747683 ~= 0.7959605

Document 2 ("cat cat fish"):
- count("cat") = 2 -> tf = 1 + ln(2) = 1.6931471805599454
  weight("cat") = 1.6931471805599454 * 1.2876820724517808 ~= 2.18023528
- count("fish") = 1 -> tf = 1 + ln(1) = 1.0
  weight("fish") = 1.0 * 1.2876820724517808 = 1.2876820724517808
- L2 norm: sqrt(2.18023528^2 + 1.2876820724517808^2)
         = sqrt(4.75342588 + 1.65812512) = sqrt(6.411551) ~= 2.532104066
- L2 normalized weights:
  v2("cat")  = 2.18023528 / 2.532104066 ~= 0.861037
  v2("fish") = 1.2876820724517808 / 2.532104066 ~= 0.508542

Query "cat":
- Normalized query vector: q("cat") = 1.0
- Cosine score(Doc 2) = 1.0 * v2("cat") ~= 0.861037
- Cosine score(Doc 1) = 1.0 * v1("cat") ~= 0.6053485
- Cosine score(Doc 3) = 0.0
"""

import math
import pytest
from pathlib import Path
from brain.index import Index, IndexedDoc
from brain.parser import ParsedNote
from brain.search import SearchEngine


@pytest.fixture
def toy_index(tmp_path: Path) -> Index:
    """Create a temporary index with the hand-verified 3-document corpus."""
    index = Index(tmp_path)

    # Document 1: "cat dog"
    n1 = ParsedNote(path="doc1.md", title="", headings=[], body="cat dog", raw_content="cat dog")
    c1 = index.compute_doc_term_counts(n1)
    index.docs["doc1.md"] = IndexedDoc(path="doc1.md", title="doc1", term_counts=c1)

    # Document 2: "cat cat fish"
    n2 = ParsedNote(path="doc2.md", title="", headings=[], body="cat cat fish", raw_content="cat cat fish")
    c2 = index.compute_doc_term_counts(n2)
    index.docs["doc2.md"] = IndexedDoc(path="doc2.md", title="doc2", term_counts=c2)

    # Document 3: "bird fish"
    n3 = ParsedNote(path="doc3.md", title="", headings=[], body="bird fish", raw_content="bird fish")
    c3 = index.compute_doc_term_counts(n3)
    index.docs["doc3.md"] = IndexedDoc(path="doc3.md", title="doc3", term_counts=c3)

    index.recompute_weights()
    return index


def test_hand_computed_idf_values(toy_index: Index):
    """Verify IDF values match exact hand calculations:

    N = 3:
    df=1 -> ln(4/2) + 1 = ln(2) + 1 ~= 1.6931471805599454
    df=2 -> ln(4/3) + 1 ~= 1.2876820724517808
    """
    expected_idf_df1 = math.log(4.0 / 2.0) + 1.0
    expected_idf_df2 = math.log(4.0 / 3.0) + 1.0

    assert pytest.approx(expected_idf_df1, rel=1e-6) == 1.6931471805599454
    assert pytest.approx(expected_idf_df2, rel=1e-6) == 1.2876820724517808

    assert pytest.approx(toy_index.idfs["dog"], rel=1e-6) == expected_idf_df1
    assert pytest.approx(toy_index.idfs["bird"], rel=1e-6) == expected_idf_df1
    assert pytest.approx(toy_index.idfs["cat"], rel=1e-6) == expected_idf_df2
    assert pytest.approx(toy_index.idfs["fish"], rel=1e-6) == expected_idf_df2


def test_hand_computed_doc_weights_and_norms(toy_index: Index):
    """Verify document weights and L2 norms match hand calculation:

    Doc 1 norm ~= 2.1271747683
    Doc 1 cat weight ~= 0.6053485
    Doc 1 dog weight ~= 0.7959605
    Doc 2 norm ~= 2.532104066
    Doc 2 cat weight ~= 0.861037
    Doc 2 fish weight ~= 0.508542
    """
    doc1 = toy_index.docs["doc1.md"]
    doc2 = toy_index.docs["doc2.md"]

    assert pytest.approx(doc1.norm, rel=1e-5) == 2.1271748
    assert pytest.approx(doc1.weights["cat"], rel=1e-5) == 0.6053485
    assert pytest.approx(doc1.weights["dog"], rel=1e-5) == 0.7959605

    assert pytest.approx(doc2.norm, rel=1e-5) == 2.5321041
    assert pytest.approx(doc2.weights["cat"], rel=1e-5) == 0.8610370
    assert pytest.approx(doc2.weights["fish"], rel=1e-5) == 0.5085420


def test_hand_computed_cosine_ranking(toy_index: Index):
    """Verify query 'cat' scores match hand calculation:

    score(doc2) ~= 0.861037
    score(doc1) ~= 0.6053485
    score(doc3) = 0.0
    """
    engine = SearchEngine(toy_index)
    results = engine.search("cat")

    assert len(results) == 2
    assert results[0].path == "doc2.md"
    assert pytest.approx(results[0].score, rel=1e-5) == 0.8610370

    assert results[1].path == "doc1.md"
    assert pytest.approx(results[1].score, rel=1e-5) == 0.6053485


def test_title_and_heading_3x_boost(tmp_path: Path):
    """Verify that title and headings receive 3x frequency boost compared to body.

    Formula:
    body 'term': count = 1 -> tf = 1 + ln(1) = 1.0
    heading 'term': count = 3 -> tf = 1 + ln(3) ~= 1 + 1.0986123 = 2.0986123
    """
    index = Index(tmp_path)

    body_note = ParsedNote(path="b.md", title="", headings=[], body="transformer", raw_content="")
    heading_note = ParsedNote(path="h.md", title="", headings=["transformer"], body="", raw_content="")

    c_body = index.compute_doc_term_counts(body_note)
    c_heading = index.compute_doc_term_counts(heading_note)

    assert c_body["transform"] == 1
    assert c_heading["transform"] == 3
