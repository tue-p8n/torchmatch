---
title: Algorithms
description: The two solver families torchmatch provides — linear assignment (one-to-one matching) and optimal transport (soft, many-to-many matching) — each with quickstarts, algorithm notes, and a full op reference.
navigation:
  title: Algorithms
---

torchmatch provides two solver families, each registered as PyTorch custom ops:

- **[Assignment](/algorithms/assignment)** — the linear assignment problem (LAP): exact
  one-to-one matching via Jonker-Volgenant, Hungarian, or Greedy.
- **[Transport](/algorithms/transport)** — the optimal transport (OT) problem: soft,
  many-to-many matching via Sinkhorn, Sinkhorn divergence, unbalanced OT, or exact EMD.

Pick by matching shape: assignment when every row must pair with exactly one column,
transport when a fractional coupling between distributions is what you need.
