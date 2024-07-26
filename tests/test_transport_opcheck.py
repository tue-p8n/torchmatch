"""torch.library.opcheck (schema + faketensor) for transport ops."""

from __future__ import annotations

import pytest
import torch

_UTILS = ("test_schema", "test_faketensor")


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_log_sinkhorn_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float32)
    a = torch.full(
        (shape[0], shape[1]),
        1.0 / shape[1],
        dtype=torch.float32,
    )
    b = torch.full(
        (shape[0], shape[2]),
        1.0 / shape[2],
        dtype=torch.float32,
    )

    torch.library.opcheck(
        torch.ops.transport.log_sinkhorn,
        (cost, 0.1, 50, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_UTILS,
    )


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_sinkhorn_divergence_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float32)
    a = torch.full(
        (shape[0], shape[1]),
        1.0 / shape[1],
        dtype=torch.float32,
    )
    b = torch.full(
        (shape[0], shape[2]),
        1.0 / shape[2],
        dtype=torch.float32,
    )

    torch.library.opcheck(
        torch.ops.transport.sinkhorn_divergence,
        (cost, 0.1, 30, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_UTILS,
    )


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_unbalanced_sinkhorn_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float32)
    a = torch.full(
        (shape[0], shape[1]),
        1.0 / shape[1],
        dtype=torch.float32,
    )
    b = torch.full(
        (shape[0], shape[2]),
        1.0 / shape[2],
        dtype=torch.float32,
    )

    torch.library.opcheck(
        torch.ops.transport.unbalanced_sinkhorn,
        (cost, 0.1, 30, 1.0, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_UTILS,
    )


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_exact_emd_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float64)
    a = torch.full(
        (shape[0], shape[1]),
        1.0 / shape[1],
        dtype=torch.float64,
    )
    b = torch.full(
        (shape[0], shape[2]),
        1.0 / shape[2],
        dtype=torch.float64,
    )

    torch.library.opcheck(
        torch.ops.transport.exact_emd,
        (cost, None, a, b),
        test_utils=_UTILS,
    )


def test_sinkhorn_samples_fwd_opcheck():
    from _triton_guard import triton_kernel_compiles

    if not triton_kernel_compiles():
        pytest.skip("CUDA + functional Triton launcher required for samples face")
    n, m, d = 16, 16, 4
    x = torch.randn(n, d, device="cuda")
    y = torch.randn(m, d, device="cuda")
    a = torch.full((n,), 1.0 / n, device="cuda")
    b = torch.full((m,), 1.0 / m, device="cuda")

    torch.library.opcheck(
        torch.ops.transport._sinkhorn_samples_fwd,
        (x, y, a, b, 0.1, 50, False, 0.0, 0.0, False, 0.0, 0.5),
        test_utils=("test_schema",),
    )
