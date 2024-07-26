"""The C++ batched (unpacked) op matches a per-problem scipy reference."""

from __future__ import annotations

import pytest
import scipy.optimize as sp
import torch


def _ref_solve(c: torch.Tensor) -> set[tuple[int, int]]:
    r, col = sp.linear_sum_assignment(c.numpy())
    return set(zip(r.tolist(), col.tolist(), strict=True))


@pytest.mark.parametrize(("batch_size", "n"), [(8, 32), (4, 128), (2, 256)])
def test_lap_jv_compact_batch_unpacked_matches_scipy(batch_size, n):
    torch.manual_seed(42)
    costs = torch.rand(batch_size, n, n, dtype=torch.float64)

    matches_t, _ur_t, _uc_t, nm_t = torch.ops.assignment.jonker_compact_batch_unpacked(
        costs,
    )

    for i in range(batch_size):
        nm = int(nm_t[i].item())
        got = {(int(matches_t[i, k, 0]), int(matches_t[i, k, 1])) for k in range(nm)}
        exp = _ref_solve(costs[i])
        assert got == exp, f"batch_size={batch_size} n={n} problem {i}: mismatch"
