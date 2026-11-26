---
title: Graph Traversal Algorithms
tags: [algorithms, computer-science, graphs]
date: 2026-02-18
---

# Graph Traversal Algorithms

Graph traversal is the process of visiting each vertex in a graph systematically.

## Fundamental Traversals

1. **Breadth-First Search (BFS)**:
   - Explores neighbors layer by layer using a First-In-First-Out (FIFO) queue.
   - Computes shortest paths on unweighted graphs.
   - Ideal for neighborhood discovery within arbitrary degree depths.

2. **Depth-First Search (DFS)**:
   - Recursively dives down paths until backtracking is required.
   - Used for cycle detection, topological sorting, and connected component discovery.

## Computational Applications

Traversals are central to dependency resolution, web crawling, and knowledge graph mapping in systems like [[inverted-index-design]] and [[dynamic-programming]].
