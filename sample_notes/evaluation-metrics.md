---
title: Evaluation Metrics for Retrieval and Classification
tags: [machine-learning, search, information-retrieval]
date: 2026-02-15
---

# Evaluation Metrics for Retrieval and Classification

Proper evaluation metrics quantify algorithm performance and prevent misleading assessments of predictive accuracy.

## Retrieval Metrics

In search engines and [[vector-space-models]], the relevance of ranked lists is measured through:
- **Precision@K**: The proportion of retrieved documents in the top $K$ results that are relevant.
- **Recall@K**: The fraction of all relevant documents captured in the top $K$.
- **Mean Average Precision (MAP)**: Measures rank-sensitive precision across multiple information retrieval queries.
- **Normalized Discounted Cumulative Gain (NDCG)**: Evaluates graded relevance where top-ranked items receive greater logarithmic weighting.

Corresponds with lexical scoring models like [[tfidf-mathematics]] to evaluate retrieval quality.
