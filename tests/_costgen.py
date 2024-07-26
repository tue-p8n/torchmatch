"""Cost-matrix generators for benchmarks.

Each distribution reflects a realistic LAP workload, so end-users can
pick an algorithm by inspecting the regime that matches their data.

- ``uniform``: cost ~ U(0, 1). Algorithmic baseline.
- ``gamma``: cost ~ Gamma(2, 0.5). Long-tail "score-like" costs common
  in ML pipelines (mode ≈ 0.5, mean ≈ 1.0, tail to ~5).
- ``iou``: ``1 - IoU(box_i, box_j)`` over random axis-aligned boxes in
  ``[0, 1]²`` with size ``[0.1, 0.4]``. Realistic for detection-to-track
  association; values concentrate near 1 with a thin mass of small
  values where boxes overlap.
- ``gated_sparse``: about 70% of cells are ``+inf`` (post-gating tracker
  outputs). To guarantee feasibility, a random permutation plants
  finite cells before masking.
- ``integer_tied``: integer costs from a small support ``{0, ..., 7}``,
  cast to the requested float dtype. Stresses tie-breaking.

All generators are deterministic given ``(n, dtype, device, seed)`` and
place the output on the requested device, so no H2D traffic biases the
timed region.
"""

from __future__ import annotations

import typing

import torch

__all__ = ["DISTRIBUTIONS", "generate"]

DISTRIBUTIONS: typing.Final[tuple[str, ...]] = (
    "uniform",
    "gamma",
    "iou",
    "gated_sparse",
    "integer_tied",
)


def _uniform(n: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(n, n, dtype=dtype, device=device, generator=g)


def _gamma(n: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    # f64 intermediates avoid catastrophic cancellation in the long tail
    # before the cast back to the requested dtype.
    g = torch.Generator(device=device).manual_seed(seed)
    u1 = torch.rand(n, n, dtype=torch.float64, device=device, generator=g)
    u2 = torch.rand(n, n, dtype=torch.float64, device=device, generator=g)
    sample = -0.5 * (torch.log1p(-u1 + 1e-12) + torch.log1p(-u2 + 1e-12))
    return sample.to(dtype)


def _iou(n: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    xy_a = torch.rand(n, 2, device=device, generator=g)
    wh_a = torch.rand(n, 2, device=device, generator=g) * 0.3 + 0.1
    box_a = torch.cat([xy_a, xy_a + wh_a], dim=1)

    xy_b = torch.rand(n, 2, device=device, generator=g)
    wh_b = torch.rand(n, 2, device=device, generator=g) * 0.3 + 0.1
    box_b = torch.cat([xy_b, xy_b + wh_b], dim=1)

    lo = torch.maximum(box_a[:, None, :2], box_b[None, :, :2])
    hi = torch.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    inter = (hi - lo).clamp_min(0).prod(-1)
    area_a = (box_a[:, 2:] - box_a[:, :2]).prod(-1)[:, None]
    area_b = (box_b[:, 2:] - box_b[:, :2]).prod(-1)[None, :]
    iou = inter / (area_a + area_b - inter + 1e-12)
    return (1.0 - iou).to(dtype)


def _gated_sparse(n: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    # Plant a guaranteed-finite permutation before masking 70% of cells
    # with +inf, so every problem stays feasible. Scaling the planted
    # cells by 0.5 ensures the optimum is not always the planted
    # permutation itself, forcing solvers to search.
    g = torch.Generator(device=device).manual_seed(seed)
    cost = torch.rand(n, n, dtype=dtype, device=device, generator=g)
    perm = torch.randperm(n, generator=g, device=device)
    rows = torch.arange(n, device=device)

    mask_prob = torch.full((n, n), 0.7, device=device)
    mask_prob[rows, perm] = 0.0  # never mask the planted permutation
    mask = torch.rand(n, n, generator=g, device=device) < mask_prob
    cost = cost.masked_fill(mask, float("inf"))

    cost[rows, perm] *= 0.5
    return cost


def _integer_tied(n: int, dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 8, (n, n), generator=g, device=device).to(dtype)


_GEN = {
    "uniform": _uniform,
    "gamma": _gamma,
    "iou": _iou,
    "gated_sparse": _gated_sparse,
    "integer_tied": _integer_tied,
}


def generate(
    dist: str,
    n: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
    batch: int | None = None,
) -> torch.Tensor:
    """Return a cost matrix (or batched stack) for the given distribution.

    ``batch=None`` returns ``(n, n)``; an integer returns ``(batch, n, n)``
    with each problem drawn from an independent generator seed.
    """
    if batch is None:
        return _GEN[dist](n, dtype, device, seed)
    return torch.stack(
        [_GEN[dist](n, dtype, device, seed + i) for i in range(batch)]
    ).contiguous()
