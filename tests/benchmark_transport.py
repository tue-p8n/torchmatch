"""pytest-benchmark suite: transport ops — matrix and samples faces.

Matrix face
-----------
Parametrised over backend, problem size N, and dtype on CPU and CUDA.
``op_name`` is the backend string accepted by ``transport.matrix.solve``:
"log_sinkhorn", "sinkhorn_divergence", "unbalanced_sinkhorn", "exact_emd".
``n_iter`` is fixed at 100 (the API default) so timings are comparable
across backends and runs.  ``exact_emd`` is skipped for N > 128 because
the network-simplex worst-case is O((N+M)^3 log(N+M)) and the run would
dominate the suite.

Samples face
------------
CUDA-only.  Parametrised over point-cloud size N, feature dimension D,
and whether the debiased Sinkhorn divergence is computed (``op_name``
"samples_loss" vs "samples_loss_debias").  The squared-Euclidean cost is
computed on the fly inside the Triton streaming kernel; no N × M matrix
is allocated.

Cost / input generation
-----------------------
Matrix face uses a uniform-random (1, N, N) cost matrix.  The transport
algorithm is agnostic to the cost distribution (unlike LAP, where
integer-tied costs change the algorithm path), so a single distribution
is sufficient.  Samples face uses unit-Gaussian point clouds in R^D.
"""

from __future__ import annotations

import pytest
import torch
import torchmatch

pytestmark = pytest.mark.timeout(0)

_CUDA = torch.cuda.is_available()
_CUDA_SKIP = pytest.mark.skipif(not _CUDA, reason="CUDA required")

_DTYPES = [torch.float32, torch.float64]
_DTYPE_IDS = ["f32", "f64"]

_SINKHORN_OPS = [
    "log_sinkhorn",
    "sinkhorn_divergence",
    "unbalanced_sinkhorn",
]
_ALL_MATRIX_OPS = [*_SINKHORN_OPS, "exact_emd"]

_MATRIX_SIZES = [64, 256, 1024]
_EMD_MAX_N = 128  # exact_emd is skipped above this threshold
_N_ITER = 100

# Samples face (CUDA only)
_SAMPLES_SIZES = [256, 1024, 4096]
_SAMPLES_DIMS = [3, 64]
_SAMPLES_OPS = ["samples_loss", "samples_loss_debias"]


# ── Input constructors ────────────────────────────────────────────────────────


def _cost(n: int, dtype: torch.dtype, device: str, seed: int = 0) -> torch.Tensor:
    """Uniform (1, N, N) cost matrix on the target device."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(1, n, n, dtype=dtype, device=device, generator=g)


def _points(n: int, d: int, device: str, seed: int = 0) -> torch.Tensor:
    """Unit-Gaussian (N, D) point cloud on the target device."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, d, device=device, generator=g)


# ── Matrix face: CPU ──────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="transport-matrix-cpu")
@pytest.mark.parametrize("op_name", _ALL_MATRIX_OPS)
@pytest.mark.parametrize("n", _MATRIX_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_matrix_cpu(benchmark, dtype, n, op_name):
    if op_name == "exact_emd" and n > _EMD_MAX_N:
        pytest.skip(f"exact_emd skipped for N={n} > {_EMD_MAX_N}")
    cost = _cost(n, dtype, "cpu")
    benchmark(
        torchmatch.transport.matrix.solve,
        cost,
        backend=op_name,
        n_iter=_N_ITER,
    )


# ── Matrix face: CUDA ─────────────────────────────────────────────────────────


@_CUDA_SKIP
@pytest.mark.benchmark(group="transport-matrix-cuda")
@pytest.mark.parametrize("op_name", _SINKHORN_OPS)
@pytest.mark.parametrize("n", _MATRIX_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_matrix_sinkhorn_cuda(benchmark, dtype, n, op_name):
    cost = _cost(n, dtype, "cuda")

    def _run():
        out = torchmatch.transport.matrix.solve(cost, backend=op_name, n_iter=_N_ITER)
        torch.cuda.synchronize()
        return out

    benchmark(_run)


# ── Samples face: CUDA ────────────────────────────────────────────────────────


@_CUDA_SKIP
@pytest.mark.benchmark(group="transport-samples-cuda")
@pytest.mark.parametrize("op_name", _SAMPLES_OPS)
@pytest.mark.parametrize("dim", _SAMPLES_DIMS)
@pytest.mark.parametrize("n", _SAMPLES_SIZES)
def test_samples_cuda(benchmark, n, dim, op_name):
    x = _points(n, dim, "cuda", seed=0)
    y = _points(n, dim, "cuda", seed=1)
    debias = op_name == "samples_loss_debias"

    def _run():
        out = torchmatch.transport.samples.loss(x, y, debias=debias)
        torch.cuda.synchronize()
        return out

    benchmark(_run)
