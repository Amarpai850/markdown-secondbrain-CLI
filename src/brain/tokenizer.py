"""Tokenizer, stopwords filter, and rule-based stemmer built from scratch."""

import re
from typing import Iterable

# Comprehensive standard English stopword list (~140 words)
STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
})

_VOWELS = frozenset("aeiouy")


def _has_vowel(s: str) -> bool:
    """Check if string contains at least one vowel."""
    return any(c in _VOWELS for c in s)


def stem_word(word: str) -> str:
    """Stem an English word using a deterministic rule-based stemmer written from scratch.

    Applies morphological suffix-stripping rules while protecting short stems and
    preserving non-inflected words.
    """
    word = word.lower()
    if len(word) <= 2:
        return word

    # Step 1: Plurals and basic inflectional endings
    if word.endswith("sses"):
        word = word[:-2]  # processes -> process
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-2]  # queries -> queri, summaries -> summari
    elif word.endswith("ices") and len(word) > 5:
        word = word[:-4] + "ic"  # matrices -> matric
    elif word.endswith("ss"):
        pass  # loss -> loss
    elif word.endswith("s") and not word.endswith("us") and not word.endswith("is"):
        stem = word[:-1]
        if len(stem) >= 2 and _has_vowel(stem) and not word.endswith("ss"):
            word = stem

    # Step 2: Past tense and participles (-eed, -ed, -ing)
    if word.endswith("eed"):
        stem = word[:-3]
        if len(stem) >= 2 and _has_vowel(stem):
            word = word[:-1]  # agreed -> agree
    elif word.endswith("ed"):
        stem = word[:-2]
        if len(stem) >= 2 and _has_vowel(stem):
            word = stem
            # double consonant reduction (stopped -> stop, trained -> train)
            if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
    elif word.endswith("ing"):
        stem = word[:-3]
        # protect words like sing, ring, king, wing (len <= 4 or no vowel in stem)
        if len(stem) >= 2 and _has_vowel(stem):
            word = stem
            # double consonant reduction (running -> run, hopping -> hop)
            if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]

    # Step 3: Derivational suffixes
    derivational = (
        ("ational", "ate"),
        ("tional", "tion"),
        ("ization", "ize"),
        ("isation", "ize"),
        ("izer", "ize"),
        ("iser", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("ability", "able"),
        ("ibility", "able"),
    )
    for suffix, replacement in derivational:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) >= 2:
                word = stem + replacement
                break

    # Step 4: Common terminal suffixes
    terminal_suffixes = (
        "ment", "ness", "able", "ible", "ance", "ence", "ant", "ent", "ism", "ist", "ize", "ise", "al"
    )
    for suffix in terminal_suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) >= 3 and _has_vowel(stem):
                word = stem
                break

    # Step 4b: -ion preceded by s or t (attention -> attent, division -> divis)
    if word.endswith("ion") and len(word) >= 5 and word[-4] in "st":
        stem = word[:-3]
        if len(stem) >= 3 and _has_vowel(stem):
            word = stem

    # Step 4c: Agentive/comparative -er suffix
    # Protected words and minimum stem length >= 4 ensure common base nouns are not mangled
    er_protected = frozenset({
        "layer", "power", "paper", "number", "computer", "order", "member",
        "water", "other", "under", "after", "over", "never", "ever", "weather",
        "danger", "matter", "letter", "corner", "river", "silver", "winter"
    })
    if word.endswith("er") and word not in er_protected:
        stem = word[:-2]
        if len(stem) >= 4 and _has_vowel(stem):
            if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]
            word = stem

    # Step 5: Adverbial -ly
    if word.endswith("ly"):
        stem = word[:-2]
        if len(stem) >= 3 and _has_vowel(stem):
            word = stem

    return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Tokenize text into lowercase, stopword-filtered, stemmed terms."""
    if not text:
        return []

    words = re.findall(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b", text.lower())
    result: list[str] = []
    for w in words:
        if w in STOPWORDS:
            continue
        # Strip trailing possessives if any ('s)
        if w.endswith("'s"):
            w = w[:-2]
        if not w or w in STOPWORDS:
            continue
        if stem:
            w = stem_word(w)
        if w:
            result.append(w)
    return result
