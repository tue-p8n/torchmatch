"""Loader for the CPU extension (dense + compact Jonker-Volgenant)."""

from __future__ import annotations

import functools
from typing import Final

import torch
from torch.utils.cpp_extension import load

from torchmatch.assignment._loader import (
    PACKAGE_DIR,
    load_extension_module,
    register_batch_row_fakes,
    register_batch_unpacked_fakes,
    register_row_fakes,
)

__all__ = ["load_cpu"]

_EXT_NAME: Final[str] = "torchmatch_assignment_cpu"
_CSRC_DIR: Final = PACKAGE_DIR / "assignment" / "cpu"

_SINGLE_OPS: Final[tuple[str, ...]] = (
    "jonker_scalar",
    "jonker_dense",
    "jonker_compact",
)
_BATCH_OPS: Final[tuple[str, ...]] = (
    "jonker_dense_batch",
    "jonker_compact_batch",
)
_BATCH_UNPACKED_OPS: Final[tuple[str, ...]] = (
    "jonker_dense_batch_unpacked",
    "jonker_compact_batch_unpacked",
)


def _simd_flags() -> list[str]:
    return ["-mavx2", "-mfma"] if torch.cpu._is_avx2_supported() else []


def _jit_build() -> None:
    load(
        name=_EXT_NAME,
        sources=[
            str(_CSRC_DIR / "jonker_scalar.cpp"),
            str(_CSRC_DIR / "jonker_dense_core.cpp"),
            str(_CSRC_DIR / "ops.cpp"),
        ],
        extra_include_paths=[str(_CSRC_DIR)],
        extra_cflags=["-O3", "-std=c++17", *_simd_flags()],
        is_python_module=False,
        verbose=False,
    )


def _register_fakes() -> None:
    register_row_fakes(_SINGLE_OPS)
    register_batch_row_fakes(_BATCH_OPS)
    register_batch_unpacked_fakes(_BATCH_UNPACKED_OPS)


@functools.cache
def load_cpu() -> None:
    """
    Register ``torch.ops.assignment.jonker_*`` (CPU backend).

    Prefers a prebuilt extension shipped in the wheel and falls back
    to JIT-compiling the C++ sources via
    :func:`torch.utils.cpp_extension.load`. Set
    ``TORCHMATCH_FORCE_JIT=1`` to skip the prebuilt path.
    """
    load_extension_module("_assignment_cpu_impl", _jit_build, _register_fakes)
