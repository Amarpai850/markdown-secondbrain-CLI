---
title: Dynamic Programming Principles
tags: [algorithms, computer-science, optimization]
date: 2026-02-26
---

# Dynamic Programming Principles

Dynamic programming solves complex problems by breaking them down into simpler subproblems and storing subproblem results to avoid redundant computations.

## Core Properties

1. **Optimal Substructure**: The optimal solution to the problem contains within it optimal solutions to subproblems.
2. **Overlapping Subproblems**: The problem can be broken down into subproblems which are reused several times.

## Implementation Paradigms

- **Top-Down with Memoization**: Solves recursively while caching evaluated subproblems in hash tables.
- **Bottom-Up with Tabulation**: Computes values iteratively in topological order, analogous to DAG scheduling in [[graph-traversal-algorithms]].
