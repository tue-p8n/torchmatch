"""torch.library.opcheck for transport ops.

The Sinkhorn ops carry an autograd formula, so they also get the autograd
registration probe and the AOT dispatch checks, which run the backward
under AOTAutograd tracing and compare it with eager. exact_emd has no
formula and keeps the schema and FakeTensor checks only.
"""

from __future__ import annotations

import pytest
import torch

_UTILS = ("test_schema", "test_faketensor")
# test_aot_dispatch_dynamic is left out: it triples the divergence op's
# runtime and brings the test within reach of the 10 s timeout; the static
# variant already runs the backward under AOTAutograd tracing.
_SINKHORN_UTILS = (
    *_UTILS,
    "test_autograd_registration",
    "test_aot_dispatch_static",
)


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_log_sinkhorn_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
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
        torch.ops.transport.log_sinkhorn,
        (cost, 0.1, 20, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_SINKHORN_UTILS,
    )


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_sinkhorn_divergence_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
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
        torch.ops.transport.sinkhorn_divergence,
        (cost, 0.1, 20, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_SINKHORN_UTILS,
    )


@pytest.mark.parametrize("shape", [(2, 3, 3), (1, 3, 5)])
def test_unbalanced_sinkhorn_opcheck(shape):
    cost = torch.rand(*shape, dtype=torch.float64, requires_grad=True)
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
        torch.ops.transport.unbalanced_sinkhorn,
        (cost, 0.1, 20, 1.0, a, b),
        kwargs={"mask": None, "scaling": None},
        test_utils=_SINKHORN_UTILS,
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
