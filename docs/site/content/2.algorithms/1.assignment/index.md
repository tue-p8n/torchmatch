---
title: Assignment
description: Linear assignment problem solvers — Jonker-Volgenant, Hungarian (Munkres, Lawler), and Greedy — registered as PyTorch custom ops.
navigation:
  title: Assignment
---

The **assignment** family solves the **linear assignment problem (LAP)**: given a cost matrix
`C` of shape `(N, M)`, find the one-to-one mapping from rows to columns that minimises the
total cost. The problem is also called bipartite matching — pairing items from one set (rows)
with items from another set (columns) so each item appears in at most one pair — or the
weighted assignment problem.

torchmatch exposes the assignment solvers through a single dispatcher:

```python
import torchmatch

# 2-D cost: single problem, CPU or CUDA
row_to_col = torchmatch.assignment.solve(cost)

# 3-D cost: batch of problems, CPU or CUDA
row_to_col = torchmatch.assignment.solve(costs)   # shape (B, N)
```

The dispatcher (`Backend.AUTO`) picks the fastest registered op for the device, shape, and
cost structure. Direct op handles (e.g., `torchmatch.assignment.ops.jonker_dense`) are also
available when you need to fix a particular algorithm — for benchmarking or to avoid the
auto-selection overhead.

## When assignment is the right tool

Use assignment when you need a **hard, one-to-one matching** between two sets of items of
known, finite size. Common situations:

- **Object tracking**: match each detected bounding box this frame to one track from the
  previous frame.
- **Set-prediction losses**: match each model prediction to one ground-truth target so you can
  compute a per-pair loss — as used in transformer-based object detectors like DETR.
- **Cluster evaluation**: when two clustering algorithms assign different integer labels to the
  same groups, find the label mapping that maximises the overlap before comparing them.
- **Replacing SciPy**: `torchmatch.assignment.solve` is a drop-in replacement for
  `scipy.optimize.linear_sum_assignment` that accepts and returns PyTorch tensors directly,
  without leaving the GPU or converting to NumPy arrays.

If you are comparing probability distributions, working with point clouds, or want a fractional
(probabilistic) rather than hard one-to-one matching, look at [Transport](/algorithms/transport) instead.

## Three solver families

| Family | Ops | Hardware |
|---|---|---|
| Jonker-Volgenant (successive-shortest-path) | `jonker_scalar`, `jonker_dense`, `jonker_compact` and their batched variants | CPU (AVX2 SIMD); CUDA for `jonker_dense_batch` |
| Hungarian algorithm family | `munkres`, `lawler`, `hybrid` | CUDA only |
| Pure-Python heuristics | `greedy`, `auction_assignment` | CPU and CUDA (no compiled extension) |

`auction_assignment` (Bertsekas' auction algorithm) returns a
`(matches, unmatched_rows, unmatched_cols)` triple and is not wired into
`solve`. Use it when you need an optimal or near-optimal match without the
compiled extension. `greedy` is a single-pass O(N²) heuristic suited for
warm-starts or profiling.

The [Algorithms](/algorithms/assignment/algorithms) page explains the mathematical differences. The
[Choosing](/algorithms/assignment/choosing) guide maps problem characteristics to the right op.

## In this section

- [Quickstart](/algorithms/assignment/quickstart) — install, import, and run your first assignment
- [Tracking tutorial](/algorithms/assignment/tracking) — end-to-end batched tracking with `jonker_dense_batch`
- [Algorithms](/algorithms/assignment/algorithms) — how Jonker-Volgenant, Munkres, and Lawler work and how they differ
- [Reference](/algorithms/assignment/reference) — full signature for every op
- [Choosing](/algorithms/assignment/choosing) — benchmark-backed decision tree
- [Tutorials](/algorithms/assignment/tutorials) — deeper notebook-based walkthroughs, from first
  principles through object tracking

See also the [history of applications](/resources/assignment-applications) — tracking, DETR losses, cluster evaluation, and more.
