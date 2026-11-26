"""Unit tests for tokenizer, stopwords filtering, and rule-based stemmer."""

import pytest
from brain.tokenizer import STOPWORDS, stem_word, tokenize


def test_stopwords_contain_common_words():
    assert "the" in STOPWORDS
    assert "is" in STOPWORDS
    assert "and" in STOPWORDS
    assert "retrieval" not in STOPWORDS
    assert "vector" not in STOPWORDS


def test_stem_word_plurals():
    assert stem_word("processes") == "process"
    assert stem_word("queries") == "queri"
    assert stem_word("vectors") == "vector"
    assert stem_word("categories") == "categori"
    assert stem_word("matrices") == "matric"
    assert stem_word("loss") == "loss"
    assert stem_word("losses") == "loss"


def test_stem_word_short_words_protected():
    assert stem_word("as") == "as"
    assert stem_word("is") == "is"
    assert stem_word("ai") == "ai"
    assert stem_word("ml") == "ml"
    assert stem_word("sing") == "sing"
    assert stem_word("ring") == "ring"
    assert stem_word("king") == "king"
    assert stem_word("red") == "red"
    assert stem_word("bed") == "bed"


def test_stem_word_past_and_participles():
    assert stem_word("connected") == "connect"
    assert stem_word("trained") == "train"
    assert stem_word("stopped") == "stop"
    assert stem_word("running") == "run"
    assert stem_word("ranking") == "rank"
    assert stem_word("learning") == "learn"


def test_stem_word_derivational_suffixes():
    assert stem_word("relational") == "relate"
    assert stem_word("conditional") == "condit"
    assert stem_word("condition") == "condit"
    assert stem_word("regularization") == "regular"
    assert stem_word("regularisation") == "regular"
    assert stem_word("regularize") == "regular"
    assert stem_word("optimizer") == "optim"
    assert stem_word("scalability") == "scal"
    assert stem_word("scalable") == "scal"
    assert stem_word("effectiveness") == "effective"


def test_tokenize_text():
    text = "The quick brown fox is running towards 10 jumping foxes!"
    tokens = tokenize(text, stem=True)
    assert "the" not in tokens
    assert "is" not in tokens
    assert "run" in tokens
    assert "fox" in tokens
    assert "jump" in tokens
    assert "10" in tokens


def test_tokenize_empty_and_punctuation():
    assert tokenize("") == []
    assert tokenize("   ... !!! ???   ") == []
    tokens = tokenize("self-attention and cross-attention")
    assert "self" in tokens
    assert "attent" in tokens
    assert "cross" in tokens


def test_stem_word_er_suffix_tightened():
    """Verify that layer, power, paper, number, computer are not mangled,
    while transformer still stems to transform.
    """
    assert stem_word("layer") == "layer"
    assert stem_word("power") == "power"
    assert stem_word("paper") == "paper"
    assert stem_word("number") == "number"
    assert stem_word("computer") == "computer"
    assert stem_word("transformer") == "transform"

