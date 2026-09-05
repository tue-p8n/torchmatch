"""
Pure-PyTorch single-pass greedy LAP heuristic (Kurtzberg 1962).

Kurtzberg, "On approximation methods for the assignment problem",
J. ACM 9(4):419-439 (1962).

At each step, pick the minimum unassigned (row, col) cell and assign
that pair, masking the row and column from further consideration.
Repeats until ``min(N, M)`` assignments have been made or the residual
matrix contains only +inf. Time complexity: O(K * N * M) where
K = min(N, M). Heuristic; never beats Jonker-Volgenant on total cost.

Registered as ``torch.ops.assignment.greedy`` via
:func:`torch.library.custom_op`, so it composes with FakeTensor and
``torch.compile`` without a separate C++ kernel.
"""

from __future__ import annotations

import torch

from torchmatch.assignment._validate import check_finite


def _greedy_2d(cost: torch.Tensor) -> torch.Tensor:
    n, m = cost.shape
    out = torch.full((n,), -1, dtype=torch.int64, device=cost.device)
    if n == 0 or m == 0:
        return out
    work = cost.clone()
    inf = torch.tensor(float("inf"), dtype=work.dtype, device=work.device)
    k = min(n, m)
    for _ in range(k):
        flat_min, flat_idx = work.flatten().min(dim=0)
        if torch.isinf(flat_min):
            break
        # Keep r and c as 0-dim tensors: int() casts force an extra
        # per-iteration device sync. The `torch.isinf(flat_min)` above is
        # itself an unavoidable host sync (data-dependent early-exit), so
        # one sync per iteration remains; the casts would double it.
        r = flat_idx // m
        c = flat_idx % m
        out[r] = c
        work[r, :] = inf
        work[:, c] = inf
    return out


@torch.library.custom_op("assignment::greedy", mutates_args=())
def greedy(cost: torch.Tensor) -> torch.Tensor:
    """
    Single-pass greedy LAP heuristic (``torch.ops.assignment.greedy``).

    Parameters
    ----------
    cost
        Cost matrix (N, M) or batch (B, N, M). float32 or float64.
        ``+inf`` marks forbidden edges (skipped). NaN and ``-inf`` are
        rejected.

    Returns
    -------
    matches
        Integer row→col assignments. Shape (N,) for 2-D input, (B, N)
        for 3-D. Unmatched rows receive ``-1``.

    """
    if cost.ndim not in (2, 3):
        msg = f"assignment.greedy: cost.ndim must be 2 or 3, got {cost.ndim}"
        raise ValueError(msg)
    # The op is a public entry point reachable as torch.ops.assignment.greedy;
    # validation cannot rely on the solve() dispatcher having run first.
    # RuntimeError mirrors what the C++ ops raise via TORCH_CHECK.
    bad = check_finite(cost)
    if bad is not None:
        msg = f"assignment.greedy: cost contains {bad}"
        raise RuntimeError(msg)
    if cost.ndim == 2:
        return _greedy_2d(cost)
    b = cost.size(0)
    if b == 0:
        # torch.stack on an empty list raises a low-level error with no op
        # context, so handle the empty-batch case symmetrically with the
        # _greedy_2d empty guard.
        return cost.new_empty(0, cost.size(1), dtype=torch.int64)
    return torch.stack([_greedy_2d(cost[i]) for i in range(b)], dim=0)


@greedy.register_fake
def _greedy_fake(cost: torch.Tensor) -> torch.Tensor:
    torch._check(cost.ndim in (2, 3), lambda: "cost must be 2D or 3D")
    if cost.ndim == 2:
        return cost.new_empty(cost.size(0), dtype=torch.int64)
    return cost.new_empty(cost.size(0), cost.size(1), dtype=torch.int64)
