---
title: Latent Semantic Analysis
tags: [information-retrieval, search, nlp, linear-algebra]
date: 2026-02-01
---

# Latent Semantic Analysis

Latent Semantic Analysis (LSA) is a technique in natural language processing that analyzes relationships between documents and terms by producing a set of concepts related to the documents and terms.

## Mathematical Foundation

LSA constructs a large term-document matrix where cell entries are weighted using [[tfidf-mathematics]]. Singular Value Decomposition (SVD) is subsequently applied to decompose the matrix:

$$A \approx U_k \Sigma_k V_k^T$$

Where:
- $U_k$ is the term-concept matrix.
- $\Sigma_k$ represents singular values scaling each latent dimension.
- $V_k^T$ is the concept-document matrix.

## Query Folding

When a search query arrives, it is projected into the low-dimensional semantic space via:

$$q_k = \Sigma_k^{-1} U_k^T q$$

This allows retrieval based on semantic proximity in [[vector-space-models]], surfacing synonymic matches that share zero lexical keywords.
