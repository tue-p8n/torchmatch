"""pytest-benchmark suite: batched ops by distribution, B, N, dtype, device."""

from __future__ import annotations

import pytest
import torch
from _costgen import DISTRIBUTIONS, generate

pytestmark = pytest.mark.timeout(0)

_CUDA = torch.cuda.is_available()
_CUDA_SKIP = pytest.mark.skipif(not _CUDA, reason="CUDA required")

_DTYPES = [torch.float32, torch.float64]
_DTYPE_IDS = ["f32", "f64"]

_CPU_OPS = [
    "jonker_dense_batch",
    "jonker_compact_batch",
    "jonker_dense_batch_unpacked",
    "jonker_compact_batch_unpacked",
]
_CPU_BATCH_SIZES = [16, 64]
_CPU_PROBLEM_SIZES = [16, 64, 128]

# CUDA tiled JV: square problems with K ≤ MAX_TILE = 64 (per jonker_tiled.cuh)
_CUDA_BATCH_SIZES = [16, 64]
_CUDA_PROBLEM_SIZES = [8, 32, 64]


@pytest.mark.benchmark(group="batch-cpu")
@pytest.mark.parametrize("op_name", _CPU_OPS)
@pytest.mark.parametrize("b", _CPU_BATCH_SIZES)
@pytest.mark.parametrize("n", _CPU_PROBLEM_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_batch_cpu(benchmark, dist, dtype, n, b, op_name):
    costs = generate(dist, n, dtype, "cpu", batch=b)
    op = getattr(torch.ops.assignment, op_name)
    benchmark(op, costs)


@_CUDA_SKIP
@pytest.mark.benchmark(group="batch-cuda")
@pytest.mark.parametrize("b", _CUDA_BATCH_SIZES)
@pytest.mark.parametrize("n", _CUDA_PROBLEM_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_batch_cuda_jv_dense(benchmark, dist, dtype, n, b):
    costs = generate(dist, n, dtype, "cuda", batch=b)

    def _run():
        out = torch.ops.assignment.jonker_dense_batch(costs)
        torch.cuda.synchronize()
        return out

    benchmark(_run)
