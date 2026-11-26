---
title: TF-IDF Mathematics and Theory
tags: [search, information-retrieval, mathematics]
date: 2026-02-22
---

# TF-IDF Mathematics and Theory

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic reflecting how important a word is to a document in a collection.

## Sublinear Term Frequency

Raw counts can overemphasize repetitive words. Sublinear scaling dampens extreme frequencies:

$$\text{tf}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{for } \text{count} > 0$$

## Smoothed Inverse Document Frequency

Inverse document frequency attenuates terms that occur universally across documents:

$$\text{idf}(t, D) = \ln\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$

## Vector Normalization

Document vectors are normalized using the Euclidean $L_2$ norm to neutralize document length bias, powering ranking in [[vector-space-models]] and semantic projections in [[latent-semantic-analysis]].
