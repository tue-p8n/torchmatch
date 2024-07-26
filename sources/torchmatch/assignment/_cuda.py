"""Loader for the CUDA extension (Hungarian solvers + tiled batched JV)."""

from __future__ import annotations

import functools
from typing import Final

import torch
from torch.utils.cpp_extension import load

from torchmatch.assignment._loader import (
    PACKAGE_DIR,
    load_extension_module,
    register_row_fakes,
)

__all__ = ["load_cuda"]

_EXT_NAME: Final[str] = "torchmatch_assignment_cuda"
_CSRC_DIR: Final = PACKAGE_DIR / "assignment" / "cuda"

_HUNGARIAN_OPS: Final[tuple[str, ...]] = (
    "munkres",
    "hybrid",
    "lawler",
)


def _jit_build() -> None:
    if not torch.cuda.is_available():
        msg = (
            "The torchmatch CUDA extension requires a CUDA-capable PyTorch "
            "build. No CUDA device was detected."
        )
        raise RuntimeError(msg)

    load(
        name=_EXT_NAME,
        sources=[
            str(_CSRC_DIR / "ops.cpp"),
            str(_CSRC_DIR / "dispatch.cu"),
            str(_CSRC_DIR / "munkres.cu"),
            str(_CSRC_DIR / "hybrid.cu"),
            str(_CSRC_DIR / "lawler.cu"),
        ],
        extra_include_paths=[str(_CSRC_DIR)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        is_python_module=False,
        verbose=False,
    )


# The FakeTensor kernel for ``jonker_dense_batch`` is registered by the
# CPU loader (it owns the shared op's schema), so the CUDA loader only
# adds the CUDA backend, not a second fake.
def _register_fakes() -> None:
    register_row_fakes(_HUNGARIAN_OPS)


@functools.cache
def load_cuda() -> None:
    """
    Register the CUDA-only ops in ``torch.ops.assignment``.

    Adds Munkres classical (``munkres``), the experimental Munkres
    hybrid (``hybrid``), Lawler tree-augmentation (``lawler``), and the
    CUDA backend of ``jonker_dense_batch``. Prefers a prebuilt extension
    shipped in the wheel and falls back to JIT-compiling the C++/CUDA
    sources via :func:`torch.utils.cpp_extension.load`. Set
    ``TORCHMATCH_FORCE_JIT=1`` to skip the prebuilt path.
    """
    load_extension_module("_assignment_cuda_impl", _jit_build, _register_fakes)
