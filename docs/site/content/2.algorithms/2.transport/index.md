---
title: Transport
description: Optimal transport solvers — Sinkhorn, Sinkhorn divergence, unbalanced OT, and exact EMD — registered as PyTorch custom ops.
navigation:
  title: Transport
---

The **transport** family solves the **optimal transport (OT) problem**: given a cost function
between two sets of points or distributions, find the minimum-cost way to move mass from one
to the other. Unlike hard one-to-one assignment (where each item is matched to exactly one other), transport handles **continuous distributions**,
**unequal total mass**, and returns **soft, differentiable plans** (matrices `P` where
`P[i,j]` is the fraction of mass moved from source `i` to target `j`; "soft" means entries
are fractional values in (0,1) rather than hard 0/1 assignments) rather than hard integer
indices (where row `i` is mapped to a single column `j`).

torchmatch exposes two surfaces:

```python
import torchmatch

# Matrix face: precomputed cost matrix in, log-plan or divergence out
log_plan = torchmatch.transport.matrix.solve(cost)        # (B, N, M)

# Samples face: raw point clouds in, scalar loss out (CUDA only)
loss = torchmatch.transport.samples.loss(x, y)            # scalar
```

## When transport is the right tool

Transport is the right tool when you need a **continuous, differentiable, or partial
matching** between distributions — when the right answer is not a hard integer index
(row `i` assigned to exactly one column `j`) but a soft plan, a distance, or a soft plan (called a coupling) that admits gradients.

### Geometric learning and 3D shape

Point clouds lack a fixed pairing: two clouds representing the same shape may have
their points in different orders, at different densities, and with noise. A naive
per-point L2 loss requires a predetermined one-to-one correspondence; the Wasserstein loss does not. `samples.loss`
computes `W_2^2` between two point sets on the fly — no cost matrix, no pre-sorting — and
gradients flow back through both sets. This makes it a natural training objective for
shape autoencoders, deformable registration networks, and any model whose output is an
unordered set of 3D coordinates.

The squared-2-Wasserstein distance also has favourable geometry: it measures how far apart two point distributions are in a way that tracks the actual spatial distances between points, so a model
minimising `W_2^2` is directly optimised for the perceptual task of producing the right
shape, not just the right pixel values.

### Soft, differentiable set-matching

Some detection models use a one-to-one matching step (called the Hungarian algorithm) to assign each predicted bounding box to one ground-truth
target. This step is non-differentiable: picking the single best match is a discrete (hard) decision, so no gradient can flow backward through that choice into the prediction network.

Replacing the Hungarian step with a Sinkhorn plan gives a **soft matching** that
interpolates between all possible pairings. Each plan entry `P_ij ∈ [0,1]` is the
probability that prediction `i` matches target `j`. The per-pair loss is a
weighted sum over the plan, and gradients flow smoothly through `P` back to the cost matrix
and hence to the network. Reducing the regularisation `ε` toward 0 sharpens the plan toward a hard one-to-one matching; raising it softens the plan toward equal weight on every possible pairing.

### Domain adaptation

A model trained on labelled source data and deployed on shifted target data can fail
because the feature distributions differ. Optimal transport provides a coupling between
source and target feature clouds that quantifies and corrects for this shift: the plan `P`
maps source samples to their closest counterparts in the target distribution (in transport
distance), which can then be used to re-weight, interpolate, or align the representations.

The Sinkhorn divergence between source and target feature distributions is also a
differentiable domain-discrepancy loss: minimising it drives the encoder to produce features
that are transport-close across domains, without needing an adversarial training loop.

### Robust partial matching

Balanced OT requires that the total weight of the source points equals the total weight of the target points (in the simplest case, both sides have the same number of equally-weighted points). In practice,
this fails when:
- Detections include false positives with no ground-truth counterpart.
- Two distributions have genuinely different total counts (e.g. the number of cells in two
  biological conditions).
- One side has outliers that should be ignored rather than forced into the coupling.

Unbalanced OT (the `UNBALANCED_SINKHORN` backend and the `reach` kwargs on `samples.loss`)
relaxes the requirement that all mass must be exactly matched, adding a penalty (called KL divergence) for any unmatched portion. Points that cannot be matched cheaply
are allowed to "disappear" — contributing to the marginal residual (the unmatched portion
of each distribution's total mass) rather than to the plan.
The `reach` parameter controls the tolerance: small `reach` is permissive; large `reach`
approaches balanced OT.

---

If the problem requires a **hard, integer-valued, one-to-one matching** with no
regularisation, look at [Assignment](/algorithms/assignment) instead.

## Two surfaces

| Surface | Input | Output | Hardware |
|---|---|---|---|
| `transport.matrix` | cost matrix `(N, M)` or `(B, N, M)` | log-plan or scalar divergence | CPU and CUDA |
| `transport.samples` | point clouds `(N, D)` and `(M, D)` | scalar Sinkhorn loss | CUDA only |

The matrix surface requires a cost matrix you construct yourself. The samples surface
computes squared-Euclidean costs on the fly inside Triton kernels — no `N × M` allocation.

## Backends

The `transport.matrix` dispatcher accepts four backends:

| Backend | Returns | Differentiable | Notes |
|---|---|---|---|
| `LOG_SINKHORN` (default) | log-plan `(B, N, M)` | yes | log-plan (the transport plan stored in log-space for numerical stability); entropic regularisation |
| `SINKHORN_DIVERGENCE` | scalar `(B,)` | yes | debiased; zero when distributions match |
| `UNBALANCED_SINKHORN` | log-plan `(B, N, M)` | yes | KL-relaxed marginals; handles outliers |
| `EXACT_EMD` | plan `(B, N, M)` | no | network simplex; exact; CPU only |

## In this section

- [Quickstart](/algorithms/transport/quickstart) — first use of `matrix.solve` and `samples.loss`
- [Point-cloud tutorial](/algorithms/transport/point-clouds) — end-to-end Wasserstein training loss
- [Algorithms](/algorithms/transport/algorithms) — Sinkhorn, debiasing, unbalanced OT, network simplex
- [Reference](/algorithms/transport/reference) — full signatures for `matrix.solve` and `samples.loss`
- [Choosing](/algorithms/transport/choosing) — which backend to use and when

See also the [history of applications](/blog/transport-applications) — generative models, domain adaptation, geometric learning, and more — on the blog.
