---
title: Regularization Techniques in Machine Learning
tags: [machine-learning, deep-learning, optimization]
date: 2026-02-10
---

# Regularization Techniques in Machine Learning

Regularization refers to modifications made to a machine learning algorithm that are intended to reduce its generalization error but not its training error.

## Key Methods

- **L2 Regularization (Ridge / Weight Decay)**: Penalizes large weights by adding $\lambda \sum w_i^2$ to the loss function during [[gradient-descent]].
- **L1 Regularization (Lasso)**: Adds $\lambda \sum |w_i|$, driving irrelevant weights to exact zero for sparse feature selection.
- **Dropout**: Randomly deactivates neurons during training passes with probability $p$, reducing co-adaptation.
- **Early Stopping**: Halts training when validation performance plateaus according to tracked [[evaluation-metrics]].
