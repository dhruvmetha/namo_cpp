---
title: "stilman-kuffner"
arxiv: ""
tags: [manipulation planning, combinatorial planning]
verdict: foundational
read: with claude
---

This work is quite foundational for the NAMO problem. Under the setting, the model of the system is known.

To solve for NAMO:
It builds an approximate graph of "free regions" in the C-space (which is effectively 2D here), edges between free regions contain the objects that disconnect them. The problem is to navigate the robot to the goal free region from a start region by minimizing the work done -- reduces pushes and use simple and efficient pushes.

They use a "Manip-Search" subroutine to find the object displacements that would connect two adjacent regions.
This subroutine is used in the "cost-efficient" full search to the goal region.
This subroutine is expensive.

This manip-search is the "region-opening" problem.