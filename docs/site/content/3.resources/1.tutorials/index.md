---
title: Tutorials
description: Interactive Jupyter notebooks covering both the assignment and transport families, from first principles through applied examples.
navigation:
  title: Tutorials
---

Interactive Jupyter notebooks for both solver families, targeting readers with
Python/PyTorch experience but no prior knowledge of combinatorial optimisation or
optimal transport.

## Assignment series

| Notebook | Topic |
|---|---|
| [The assignment problem](/resources/tutorials/assignment/01_the_assignment_problem) | Problem definition, brute-force impossibility, `assignment.solve` |
| [Backends and batching](/resources/tutorials/assignment/02_backends_and_batching) | Backend selection, batched 3-D solve, unpacked output |
| [Object tracking](/resources/tutorials/assignment/03_object_tracking) | IoU cost matrix, SORT-style tracker, trajectory visualisation |

## Transport series

| Notebook | Topic |
|---|---|
| [Optimal transport](/resources/tutorials/transport/01_optimal_transport) | Earth-mover intuition, transport plans, Wasserstein distance |
| [Sinkhorn algorithm](/resources/tutorials/transport/02_sinkhorn_algorithm) | Entropic regularisation, Sinkhorn iterations, divergence |
| [Point clouds and shapes](/resources/tutorials/transport/03_point_clouds) | `samples.loss`, shape generation, unbalanced OT |

Each notebook is also available as a written walkthrough under
[Assignment tutorials](/algorithms/assignment/tutorials) and
[Transport tutorials](/algorithms/transport/tutorials).
