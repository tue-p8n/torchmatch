---
seo:
  title: torchmatch — Assignment & Transport for PyTorch
  description: Linear assignment and optimal transport solvers for PyTorch. Tracking-by-detection, DETR-style set prediction losses, cluster relabelling, Sinkhorn OT, and point-cloud Wasserstein loss — all batched, torch.compile-ready.
---

::landing-hero
::

::landing-section
---
index: '01'
title: Use cases
subtitle: Assignment covers discrete one-to-one matching; transport covers continuous distributions and point clouds.
---

::landing-family-label
---
label: Assignment
---
::

::landing-use-case
---
index: '01'
domain: Object tracking
systems: SORT · ByteTrack · BoT-SORT · DeepSORT
title: Detection-to-track association
related:
  - label: Batched tracking tutorial
    to: /algorithms/assignment/tracking
  - label: jonker_dense_batch reference
    to: /algorithms/assignment/reference
---

#prose
A tracker reads detections each frame and assigns each one to a track. Build the per-frame `(N_tracks × M_detections)` cost matrix — typically `1 − IoU` (intersection-over-union overlap) between predicted and detected bounding boxes, optionally filtered by distance between box centers (centroid gating) or visual similarity (appearance distance) — stack frames into a batch, and `jonker_dense_batch` solves the whole batch in one CUDA launch.

The `_unpacked` variant returns `(matches, unmatched_tracks, unmatched_dets, n_matched)` directly — removing the per-frame Python loop that most codebases use to separate matched from unmatched indices after getting the raw assignment.

#code
  :::code-tabs{default="match" direct-label="jonker_dense_batch_unpacked"}
  #match
  ```python
  import torch
  import torchmatch

  # costs: (B, N_tracks, M_dets) of `1 - IoU`,
  # with +inf where centroid distance > gate.
  costs = build_iou_cost_batch(
      tracks, detections, gate=0.3,
  )                                          # (B, N, M)

  matches, ur, uc, n_matched = torchmatch.assignment.solve(
      costs, unpack=True,
  )
  # matches[b, :n_matched[b]] = (track_idx, det_idx) pairs
  ```

  #direct
  ```python
  import torch
  import torchmatch

  costs = build_iou_cost_batch(
      tracks, detections, gate=0.3,
  )                                          # (B, N, M)

  # AUTO would pick jonker_*_batch_unpacked; calling it directly lets
  # you choose dense (rectangular) vs compact (square AVX2-gather).
  out = torchmatch.assignment.ops.jonker_dense_batch_unpacked(costs)
  matches, ur, uc, n_matched = out
  ```
  :::
::

::landing-use-case
---
index: '02'
domain: Set prediction
systems: DETR · MaskFormer · Mask2Former · RT-DETR
title: Hungarian matcher for the loss
related:
  - label: Single-problem tutorial
    to: /algorithms/assignment/quickstart
  - label: Choosing the right op
    to: /algorithms/assignment/choosing
---

#prose
DETR-style object detectors find the lowest-cost one-to-one pairing (the Hungarian assignment) between model predictions and ground-truth targets before computing the per-pair loss term. The matcher runs once per image every training step, so it dominates training wall time when called naively in a Python loop. Batch the per-image cost matrices into one `(B, N_pred, N_gt)` tensor; `jonker_dense_batch` solves them all at once.

When every problem is square (prediction count fixed, GT padded to the same count), `jonker_compact_batch` runs the tighter AVX2-gather inner loop on CPU. For mixed shapes, use `jonker_dense_batch`.

#code
  :::code-tabs{default="match" direct-label="jonker_dense_batch"}
  #match
  ```python
  import torch
  import torchmatch

  # Combined L1 + GIoU + class cost between every
  # prediction and every ground-truth target, padded
  # to a common N_gt with +inf in padded columns.
  costs = compute_match_cost(preds, targets)
  # → (B, N_pred, N_gt + N_pad)

  matches = torchmatch.assignment.solve(costs)
  # matches[b, i] == -1 marks pad-column hits;
  # mask them out of the per-pair loss.
  ```

  #direct
  ```python
  import torch
  import torchmatch

  costs = compute_match_cost(preds, targets)

  # When every problem is square, jonker_compact_batch runs the tighter
  # AVX2-gather inner loop on CPU. For mixed shapes, use jonker_dense_batch.
  matches = torchmatch.assignment.ops.jonker_dense_batch(costs)
  ```
  :::
::

::landing-use-case
---
index: '03'
domain: Evaluation
systems: Unsupervised semseg · ARI · ReID · Codebook alignment
title: Cluster label matching
related:
  - label: jonker_compact details
    to: /algorithms/assignment/reference
  - label: Algorithms overview
    to: /algorithms/assignment/algorithms
---

#prose
Clustering algorithms assign arbitrary label numbers, so "cluster 0" in one run may correspond to "cluster 2" in another. Metrics like Adjusted Rand Index therefore need to find the optimal one-to-one relabelling between predicted and ground-truth cluster IDs before counting agreements — that relabelling is exactly the assignment problem. Build the cost as the negative confusion-matrix entry; the linear assignment solver (LAP) returns the optimal cluster-to-cluster mapping.

Square dense costs at moderate `K` (≤ 256) are the regime `jonker_compact` was built for. For `K ≥ 512`, `jonker_dense` takes over.

#code
  :::code-tabs{default="match" direct-label="jonker_compact"}
  #match
  ```python
  import torch
  import torchmatch

  # K x K confusion-style matrix. Negate so the LAP
  # minimizes disagreement instead of maximizing it.
  cost = -confusion_matrix.to(torch.float32)
  mapping = torchmatch.assignment.solve(cost)               # (K,)

  relabelled = mapping[predicted_labels]
  accuracy = (relabelled == ground_truth).float().mean()
  ```

  #direct
  ```python
  import torch
  import torchmatch

  cost = -confusion_matrix.to(torch.float32)

  # Square dense costs at moderate K (≤ 256) are the regime
  # jonker_compact was built for. For K ≥ 512, jonker_dense takes over.
  mapping = torchmatch.assignment.ops.jonker_compact(cost)      # (K,)
  relabelled = mapping[predicted_labels]
  ```
  :::
::

::landing-use-case
---
index: '04'
domain: Drop-in
systems: scipy.optimize.linear_sum_assignment · Hungarian solvers
title: A faster scipy LAP
related:
  - label: Single-problem tutorial
    to: /algorithms/assignment/quickstart
  - label: Benchmarks
    to: /resources/benchmarks
---

#prose
When a linear assignment problem (LAP) appears in a tight Python loop and SciPy is the bottleneck, swap the call. `jonker_dense` produces the same optimal cost, accepts the same rectangular input, and stays inside `torch` so the cost matrix never round-trips through `numpy`. For a batch of independent LAPs, `jonker_dense_batch` runs `at::parallel_for` across problems on CPU or the tiled CUDA kernel on GPU.

#code
  :::code-tabs{default="match" direct-label="jonker_dense"}
  #match
  ```python
  import torch
  import torchmatch

  # Before:
  #   from scipy.optimize import linear_sum_assignment
  #   r, c = linear_sum_assignment(cost.cpu().numpy())

  # After: stays in torch, batches naturally.
  row_to_col = torchmatch.assignment.solve(cost)
  matched = row_to_col >= 0
  total = cost[
      torch.arange(cost.size(0))[matched],
      row_to_col[matched],
  ].sum()
  ```

  #direct
  ```python
  import torch
  import torchmatch

  # jonker_dense is the rectangular-capable CPU op; AUTO routes here
  # for any non-tiny CPU problem.
  row_to_col = torchmatch.assignment.ops.jonker_dense(cost)
  matched = row_to_col >= 0
  total = cost[
      torch.arange(cost.size(0))[matched],
      row_to_col[matched],
  ].sum()
  ```
  :::
::

::landing-family-label
---
label: Optimal transport
---
::

::landing-use-case
---
index: '01'
domain: Geometric learning
systems: PointNet · ShapeFlow · 3D-LFM · Wasserstein AE
title: Point-cloud Wasserstein loss
related:
  - label: Transport ops reference
    to: /algorithms/transport/reference
---

#prose
For training generative models over point sets, `transport.samples.loss(x, y)` computes the Sinkhorn approximation of the Wasserstein distance between two point clouds — a measure of how much work it takes to move one distribution onto the other — without ever building the full N×M pairwise-cost matrix in memory. A Triton streaming kernel handles the computation directly in fast CUDA registers. Gradients flow analytically through both `x` and `y`; pass `debias=True` for the symmetric Sinkhorn divergence variant.

#code
  :::code-tabs{default="match" direct-label="samples.loss"}
  #match
  ```python
  import torch
  import torchmatch

  # predicted and ground-truth 3-D point clouds
  pred = model(z)                                 # (N, 3)
  gt   = target_cloud.to(pred.device)             # (M, 3)

  loss = torchmatch.transport.samples.loss(pred, gt)
  loss.backward()
  ```

  #direct
  ```python
  import torch
  import torchmatch

  pred = model(z)
  gt   = target_cloud.to(pred.device)

  # debias=True gives the Sinkhorn divergence variant:
  # symmetric, positive, corrects for self-transport.
  loss = torchmatch.transport.samples.loss(
      pred, gt, debias=True,
  )
  loss.backward()
  ```
  :::
::

::landing-use-case
---
index: '02'
domain: Set prediction / training
systems: Slot Attention · Soft-DETR · Differentiable permutations
title: Differentiable soft matching
related:
  - label: Transport ops reference
    to: /algorithms/transport/reference
---

#prose
Hard one-to-one matching (like the Hungarian algorithm) is not differentiable — gradients cannot flow back through a discrete matching step. Replace it with a Sinkhorn plan: `transport.matrix.solve(cost)` returns a regularized transport plan T ∈ [0,1]^{N×M} — fractional soft assignments stored in log-domain for numerical stability — that differentiate smoothly through the cost matrix. Reduce the regularization (`reg`) toward zero to sharpen the plan toward a hard permutation. Both 2-D `(N, M)` and batched 3-D `(B, N, M)` cost tensors are accepted.

#code
  :::code-tabs{default="match" direct-label="ops.log_sinkhorn"}
  #match
  ```python
  import torch
  import torchmatch
  from torchmatch.transport.matrix import Backend

  # (B, N, M) cost: negative cosine or L2 similarity
  cost = -torch.einsum("bnd,bmd->bnm", pred_feats, gt_feats)

  # log-plan (B, N, M); differentiable w.r.t. cost
  log_plan = torchmatch.transport.matrix.solve(
      cost,
      backend=Backend.LOG_SINKHORN,
  )
  loss = (log_plan.exp() * cost).sum(-1).mean()
  loss.backward()
  ```

  #direct
  ```python
  import torch
  import torchmatch

  cost = -torch.einsum("bnd,bmd->bnm", pred_feats, gt_feats)

  # Call the op directly to pin reg and n_iter.
  log_plan = torchmatch.transport.matrix.ops.log_sinkhorn(
      cost, reg=0.05, n_iter=50,
  )
  ```
  :::
::

::landing-use-case
---
index: '03'
domain: Domain adaptation / robust OT
systems: Domain shift · Partial shape matching · Noisy labels
title: Unbalanced transport
related:
  - label: Transport ops reference
    to: /algorithms/transport/reference
---

#prose
Standard optimal transport requires the source and target distributions to have exactly equal total mass — every point must be fully accounted for. If one set contains outliers or the two domains differ in size, this constraint forces those outliers into the plan and corrupts the result. `UNBALANCED_SINKHORN` relaxes the marginal constraints via a KL-divergence penalty controlled by the `reach` (or `reach_x`, `reach_y`) parameter. Smaller `reach` is more lenient; `reach → ∞` recovers balanced OT. Both the matrix and samples faces support unbalanced mode.

#code
  :::code-tabs{default="match" direct-label="samples.loss (reach)"}
  #match
  ```python
  import torch
  import torchmatch
  from torchmatch.transport.matrix import Backend

  cost = compute_cost(source_features, target_features)

  # reach controls the KL marginal penalty; smaller = more lenient
  plan = torchmatch.transport.matrix.solve(
      cost,
      backend=Backend.UNBALANCED_SINKHORN,
      reach=0.5,
  )
  ```

  #direct
  ```python
  import torch
  import torchmatch

  x = source_pts.cuda()
  y = target_pts.cuda()

  # Point-cloud unbalanced OT via samples face
  loss = torchmatch.transport.samples.loss(
      x, y,
      reach=0.5,      # or reach_x / reach_y for asymmetric
  )
  ```
  :::
::

::

::landing-section
---
index: '02'
title: What AUTO picks
subtitle: 'The assignment dispatcher routes by device, shape, and size. Seven rows of the decision tree; the full version is in the choosing tutorial.'
---

::landing-chooser
---
rows:
  - have: 'One cost matrix on CPU · <code>N×M ≤ 64</code>'
    use: 'jonker_scalar'
    why: 'Sequential reference; lowest overhead at tiny sizes'
  - have: 'One cost matrix on CPU · square · any size'
    use: 'jonker_compact'
    why: 'Tightest AVX2-gather inner loop'
  - have: 'One cost matrix on CPU · rectangular · any size'
    use: 'jonker_dense'
    why: 'AVX2 flat-pointer; rectangular-capable'
  - have: 'A batch of cost matrices on CPU'
    use: 'jonker_compact_batch <span style="color:var(--ui-color-text-dimmed)">·or·</span> jonker_dense_batch'
    why: '<code>at::parallel_for</code> over problems; compact when square'
  - have: 'A batch of square <code>K ≤ 64</code> matrices on CUDA'
    use: 'jonker_dense_batch (CUDA)'
    why: 'Single-block-per-problem tiled kernel; CUDA-graph-safe'
  - have: 'One cost matrix on CUDA · <code>N ≥ 32</code>'
    use: 'lawler'
    why: "Lawler's parallel-BFS tree augmentation; dense-favored"
  - have: 'One cost matrix on CUDA · <code>N &lt; 32</code> or tied / quantized'
    use: 'munkres'
    why: "Munkres' single-path Hungarian; sparse-favored"
---
::

[Full decision tree →](/algorithms/assignment/choosing){.lc-link}

::

::landing-section
---
index: '03'
title: Transport backends
subtitle: 'The transport dispatcher resolves by problem type. Pick the backend that matches your cost representation, marginal constraints, and differentiability needs.'
---

::landing-chooser
---
rows:
  - have: 'Differentiable soft plan from cost matrix'
    use: 'LOG_SINKHORN'
    why: 'Default AUTO; Sinkhorn LSE loop; returns log-plan (B, N, M); grads w.r.t. cost'
  - have: 'Symmetric OT metric / training loss'
    use: 'SINKHORN_DIVERGENCE'
    why: 'Debiased; positive; symmetric; cancels self-transport bias; returns scalar per batch'
  - have: 'Partial matching or unequal total mass'
    use: 'UNBALANCED_SINKHORN'
    why: 'KL-relaxed marginal constraints; tune via <code>reach</code>; handles outliers'
  - have: 'Two raw point clouds on CUDA'
    use: 'transport.samples.loss'
    why: 'Triton streaming kernel; no N×M allocation; fuses cost + LSE; grads through both sets'
  - have: 'Exact Earth Mover&apos;s Distance (no regularization)'
    use: 'EXACT_EMD'
    why: 'Network simplex; no bias; CPU-only; reference quality for small problems'
---
::

[Transport reference →](/algorithms/transport/reference){.lc-link}

::

::landing-cta
---
command: '$ pip install torchmatch'
links:
  - label: 'Quickstart →'
    href: /getting-started
  - label: Tutorials
    href: /algorithms
  - label: 'Source ↗'
    href: https://github.com/khwstolle/torchmatch
    external: true
specs:
  - key: REQUIRES
    value: 'Python 3.13 · PyTorch ≥ 2.11 · x86-64 Linux'
  - key: CUDA WHEELS
    value: 'cu126 · cu128 · cu130'
---
::
