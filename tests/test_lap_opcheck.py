r"""``torch.library.opcheck`` tests for the LAP custom ops.

These tests catch schema / fake-impl drift early, for example when
the C++ implementation grows a new output, mutates an input, or the
registered fake kernel returns a different shape from the real
kernel. They replace hand-rolled tracing-compatibility tests for
custom ops.
"""

from __future__ import annotations

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_SKIP = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")


# ``test_aot_dispatch_dynamic`` is deliberately omitted: it compares
# eager vs AOT-traced outputs by deep equality, but LAP problems with
# tied costs admit multiple optimal assignments, so the two runs may
# legally return different (equally-correct) row→col mappings. Schema
# and fake-kernel checks remain the meaningful guards against author
# error.
_OPCHECK_UTILS = ("test_schema", "test_faketensor")


# Module-scope: opcheck reads these tensors but never mutates them, so
# one allocation per session is enough. Function-scoped fixtures would
# impose redundant global-RNG churn and tensor allocation on every
# parametrized op.
@pytest.fixture(scope="module")
def cost_cuda() -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    return torch.rand(16, 16, dtype=torch.float32, device="cuda", generator=g)


@pytest.fixture(scope="module")
def cost_cuda_batch() -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(0)
    return torch.rand(4, 16, 16, dtype=torch.float32, device="cuda", generator=g)


@pytest.fixture(scope="module")
def cost_cpu() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(16, 16, dtype=torch.float32, generator=g)


@pytest.fixture(scope="module")
def cost_cpu_batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(4, 16, 16, dtype=torch.float32, generator=g)


@_CUDA_SKIP
@pytest.mark.parametrize(
    "op_name",
    ["munkres", "hybrid", "lawler"],
)
def test_hungarian_opcheck(cost_cuda, op_name):
    op = getattr(torch.ops.assignment, op_name)
    torch.library.opcheck(op, (cost_cuda,), test_utils=_OPCHECK_UTILS)


@_CUDA_SKIP
def test_lap_jv_dense_batch_cuda_opcheck(cost_cuda_batch):
    torch.library.opcheck(
        torch.ops.assignment.jonker_dense_batch,
        (cost_cuda_batch,),
        test_utils=_OPCHECK_UTILS,
    )


@pytest.mark.parametrize("op_name", ["jonker_scalar", "jonker_dense", "jonker_compact"])
def test_cpu_single_opcheck(cost_cpu, op_name):
    op = getattr(torch.ops.assignment, op_name)
    torch.library.opcheck(op, (cost_cpu,), test_utils=_OPCHECK_UTILS)


@pytest.mark.parametrize(
    "op_name",
    [
        "jonker_dense_batch",
        "jonker_compact_batch",
        "jonker_dense_batch_unpacked",
        "jonker_compact_batch_unpacked",
    ],
)
def test_lap_jv_batch_opcheck(cost_cpu_batch, op_name):
    op = getattr(torch.ops.assignment, op_name)
    torch.library.opcheck(op, (cost_cpu_batch,), test_utils=_OPCHECK_UTILS)
