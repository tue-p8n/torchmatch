"""Float32 inputs produce optimal (or near-optimal) assignments."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("n", [8, 32, 64, 128, 256])
@pytest.mark.parametrize("op_name", ["jonker_dense", "jonker_compact"])
def test_lap_jv_float32_optimal(ref_cost, n, op_name):
    torch.manual_seed(0)
    cost_f32 = torch.rand(n, n, dtype=torch.float32)
    cost_f64 = cost_f32.to(torch.float64)

    op = getattr(torch.ops.assignment, op_name)
    row_to_col = op(cost_f32)

    rows = torch.arange(n)
    matched = row_to_col >= 0
    got = float(cost_f64[rows[matched], row_to_col[matched]].sum())

    ref = ref_cost(cost_f64)
    assert abs(got - ref) / max(ref, 1e-9) < 0.01
