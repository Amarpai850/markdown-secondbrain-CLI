---
title: BM25 Ranking Function
tags: [information-retrieval, search, ranking]
date: 2026-03-14
---

# BM25 Ranking Function

BM25 (Best Matching 25) is a probabilistic ranking function used by many production search engines. It refines classic TF-IDF weighting from [[tfidf-mathematics]] with two practical corrections.

## Term Frequency Saturation

In plain TF-IDF a term that appears fifty times keeps adding score. BM25 saturates term frequency with the parameter `k1`, so the tenth occurrence of a word adds far less than the second. Typical values of k1 sit between 1.2 and 2.0.

## Document Length Normalization

Long documents mention more words by chance. The parameter `b` controls how strongly scores are normalized by document length relative to the average length of the collection. Setting b to zero disables length normalization entirely.

## Relation to the Vector Space Model

BM25 is not a cosine similarity over vectors, unlike [[vector-space-models]], but it still relies on an [[inverted-index-design]] to fetch postings for each query term efficiently. Comparing BM25 against TF-IDF on the same queries is a useful exercise in [[evaluation-metrics]] such as precision and recall.

#information-retrieval
