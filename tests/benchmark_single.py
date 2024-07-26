"""pytest-benchmark suite: single-problem ops by distribution, N, dtype, device."""

from __future__ import annotations

import pytest
import torch
from _costgen import DISTRIBUTIONS, generate

pytestmark = pytest.mark.timeout(0)

_CUDA = torch.cuda.is_available()
_CUDA_SKIP = pytest.mark.skipif(not _CUDA, reason="CUDA required")

_DTYPES = [torch.float32, torch.float64]
_DTYPE_IDS = ["f32", "f64"]
_SIZES = [16, 64, 256, 1024]
_CPU_OPS = ["jonker_scalar", "jonker_dense", "jonker_compact"]
_CUDA_OPS = ["munkres", "lawler"]


@pytest.mark.benchmark(group="single-cpu")
@pytest.mark.parametrize("op_name", _CPU_OPS)
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_single_cpu(benchmark, dist, dtype, n, op_name):
    cost = generate(dist, n, dtype, "cpu")
    op = getattr(torch.ops.assignment, op_name)
    benchmark(op, cost)


@_CUDA_SKIP
@pytest.mark.benchmark(group="single-cuda")
@pytest.mark.parametrize("op_name", _CUDA_OPS)
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_single_cuda(benchmark, dist, dtype, n, op_name):
    cost = generate(dist, n, dtype, "cuda")
    op = getattr(torch.ops.assignment, op_name)

    def _run():
        out = op(cost)
        torch.cuda.synchronize()
        return out

    benchmark(_run)
