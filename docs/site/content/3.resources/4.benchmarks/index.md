---
title: Benchmarks
description: Per-op benchmark sweep across problem sizes, dtypes, and devices — covering assignment (single-problem, batched) and transport (matrix-face Sinkhorn/EMD, samples-face Triton) ops.
---

This page aggregates pytest-benchmark runs contributed by users. The
suite covers three files:

- `tests/benchmark_single.py` — single-problem assignment ops
- `tests/benchmark_batched.py` — batched assignment ops
- `tests/benchmark_transport.py` — transport ops (matrix and samples faces)

Numbers differ across hardware. Rankings within a machine are stable;
absolute timings are hardware-specific.

To submit your own run, see [Contributing benchmarks](/resources/benchmarks/contributing).

## Overview

::bench-explorer
::

## Assignment methodology

The assignment suite parametrizes over:

- **Cost distributions**: `uniform`, `gamma`, `iou`, `gated_sparse`,
  `integer_tied`. See `tests/_costgen.py` for the generators.
- **Problem sizes**: N ∈ {16, 64, 256, 1024} for single-problem ops;
  N ∈ {16, 64, 128} for batched CPU; N ∈ {8, 32, 64} for batched CUDA
  (the tiled kernel caps at K = 64 (K is the padded square problem size; the CUDA kernel uses shared memory sized for at most 64 rows and 64 columns)).
- **Batch sizes**: B ∈ {16, 64} for batched ops.
- **Dtypes**: float32 and float64.
- **Devices**: CPU for the Jonker-Volgenant (JV) family (`jonker_scalar`, `jonker_dense`, `jonker_compact`, and the CPU backend of `jonker_dense_batch`); CUDA for `munkres`, `lawler`, and the CUDA backend of `jonker_dense_batch`.

## Transport methodology

The transport suite parametrizes over:

- **Backends** (matrix face): `log_sinkhorn`, `sinkhorn_divergence`,
  `unbalanced_sinkhorn`, `exact_emd`. Each backend receives a uniform
  random cost matrix; the transport algorithm does not branch on cost
  distribution, so one distribution is sufficient.
- **Problem sizes** (matrix face): N ∈ {64, 256, 1024}. `exact_emd` is
  skipped above N = 128 (network-simplex worst case is O((N+M)³ log(N+M))).
- **Point-cloud size** (samples face): N ∈ {256, 1024, 4096} points.
- **Feature dimension** (samples face): D ∈ {3, 64} — 3D shape coordinates
  and feature-space clouds.
- **Modes** (samples face): `samples_loss` (standard Sinkhorn, one solver call) and
  `samples_loss_debias` (Sinkhorn divergence, which removes entropic bias by combining three Sinkhorn solves).
- **Dtypes** (matrix face): float32 and float64. The samples face operates
  in float32 (the Triton kernel precision).
- `n_iter` is fixed at 100 for all Sinkhorn backends.

All timed cases run through `pytest-benchmark` (warm-up + multiple rounds,
median reported). CUDA cases synchronize the stream inside the timed region.

## Distributions tested

| Name | Description | Why it matters |
|---|---|---|
| `uniform` | `cost ~ U(0, 1)` | Algorithmic baseline |
| `gamma` | `cost ~ Gamma(2, 0.5)` (long tail to ~5) | Score-like ML pipeline costs |
| `iou` | `1 - IoU(box_i, box_j)` for random axis-aligned bounding boxes (AABBs) | Realistic for assigning detected objects to tracked objects across frames |
| `gated_sparse` | ~70 % `+inf` (forbidden pairs), with at least one feasible perfect matching guaranteed | Models tracker outputs after a distance-gating step rejects implausible pairings |
| `integer_tied` | Random integers in `{0, ..., 7}` cast to float | Stresses tie-breaking; quantized cost workloads |

## Single-problem CPU latency

Median across `uniform` cost on float32. Use the machine picker above
to restrict the dataset to one machine.

::bench-chart{group="single-cpu" x="n" series="op" filter='{"dtype":"f32","dist":"uniform"}' title="single-cpu, f32, uniform"}
::

::bench-table{group="single-cpu" row="n" col="op" filter='{"dtype":"f32","dist":"uniform"}' title="single-cpu, f32, uniform (median)"}
::

## Single-problem CUDA latency

::bench-chart{group="single-cuda" x="n" series="op" filter='{"dtype":"f32","dist":"uniform"}' title="single-cuda, f32, uniform"}
::

::bench-table{group="single-cuda" row="n" col="op" filter='{"dtype":"f32","dist":"uniform"}' title="single-cuda, f32, uniform (median)"}
::

## Batched CPU latency

::bench-chart{group="batch-cpu" x="n" series="op" filter='{"b":16,"dtype":"f32","dist":"uniform"}' title="batch-cpu, B=16, f32, uniform"}
::

::bench-table{group="batch-cpu" row="n" col="op" filter='{"b":16,"dtype":"f32","dist":"uniform"}' title="batch-cpu, B=16, f32, uniform (median)"}
::

## Batched CUDA latency

::bench-chart{group="batch-cuda" x="n" series="b" filter='{"dtype":"f32","dist":"uniform"}' title="batch-cuda jonker_dense_batch, f32, uniform"}
::

::bench-table{group="batch-cuda" row="n" col="b" filter='{"dtype":"f32","dist":"uniform"}' title="batch-cuda, f32, uniform (median, B columns)"}
::

## Transport matrix-face CPU latency

Median across float32, uniform cost. Compare Sinkhorn backends across
problem sizes; `exact_emd` only appears at N ≤ 128.

::bench-chart{group="transport-matrix-cpu" x="n" series="op" filter='{"dtype":"f32"}' title="transport matrix CPU, f32"}
::

::bench-table{group="transport-matrix-cpu" row="n" col="op" filter='{"dtype":"f32"}' title="transport matrix CPU, f32 (median)"}
::

## Transport matrix-face CUDA latency

::bench-chart{group="transport-matrix-cuda" x="n" series="op" filter='{"dtype":"f32"}' title="transport matrix CUDA, f32"}
::

::bench-table{group="transport-matrix-cuda" row="n" col="op" filter='{"dtype":"f32"}' title="transport matrix CUDA, f32 (median)"}
::

## Transport samples-face CUDA latency

Point-cloud optimal transport (OT) via the Triton streaming kernel. D = 3 (3D shapes) and
D = 64 (feature-space clouds). `samples_loss_debias` runs three forward
passes; the ratio to `samples_loss` is roughly 3× for large N but
slightly more at small N due to fixed overhead.

::bench-chart{group="transport-samples-cuda" x="n" series="op" filter='{"dim":3}' title="transport samples CUDA, D=3"}
::

::bench-chart{group="transport-samples-cuda" x="n" series="op" filter='{"dim":64}' title="transport samples CUDA, D=64"}
::

## Assignment caveats

The assignment sweep covers common Linear Assignment Problem (LAP) cost regimes but not:

- Mahalanobis-distance costs (covariance-weighted distances produced by Kalman-filter trackers, common in SORT-style tracking).
- Cosine-similarity costs in high-dimensional re-identification (re-ID) embeddings.
- Truly sparse problems (≥ 95 % `+inf`).
- Very small problems (N = 2 to 8), where launch overhead may flip the ranking.

Extend `tests/_costgen.py` and re-run for out-of-distribution workloads. See
[Choosing the right op](/algorithms/assignment/choosing) for the decision tree.

## Transport caveats

The transport sweep uses uniform random cost matrices. Sinkhorn convergence
speed depends on cost structure: smooth, low-variance costs converge faster
(fewer effective iterations) than costs with a very wide range of values or near-uniform rows/columns, which require more iterations to reach a good transport plan.
The 100-iteration fixed `n_iter` is conservative for typical use. See
[Choosing the right backend](/algorithms/transport/choosing) for guidance.
