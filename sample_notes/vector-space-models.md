---
title: Vector Space Models for Search
tags: [search, information-retrieval, linear-algebra]
date: 2026-02-24
---

# Vector Space Models for Search

The Vector Space Model represents text documents as vectors in a multi-dimensional Euclidean space whose axes correspond to vocabulary terms.

## Cosine Similarity

Relevance between a query vector $\mathbf{q}$ and a document vector $\mathbf{d}$ is computed as the cosine of the angle between them:

$$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} = \sum_{t \in q \cap d} q_t \cdot d_t$$

When vectors are pre-normalized to unit $L_2$ length, cosine similarity reduces to a fast dot product over non-zero posting terms stored in an [[inverted-index-design]].

See related theoretical foundations in [[tfidf-mathematics]] and [[evaluation-metrics]].
