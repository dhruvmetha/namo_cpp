---
uid: "613xz839"
title: "Learning to Rank for Synthesizing Planning Heuristics"
arxiv: "1608.01302"
authors: "Caelan Reed Garrett, Leslie Pack Kaelbling, Tomás Lozano-Pérez"
venue: "IJCAI 2016"
tags: ["learning-to-rank", "heuristic-learning", "planning-search", "ranksvm"]
verdict: "QUEUE-WORTHY — directly addresses learned ranking to guide search with oracle verification"
read: false
projects: "namo"
slug: "1608.01302"
---

RankSVM approach to learning planning heuristics by optimizing ranking of states rather than regression to absolute heuristic values. Key insight: in greedy best-first search, the ordering induced by a heuristic is more important to planning success than mean squared error. Introduces pairwise-action features capturing temporal interactions in approximate plans.

Experiments on IPC 2014 learning track domains (elevators, transport, parking, no-mystery) show RankSVM learned heuristics substantially outperform ridge regression baseline in coverage and search performance.

Relevance: Matches criterion (b) — learned ranking directly guides search (greedy best-first), and oracle (the planner's simulator) verifies candidate solutions. The ranker's job is ordering pushes the search tries first, which is exactly our use case.

