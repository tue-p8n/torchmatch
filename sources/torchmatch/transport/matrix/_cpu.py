"""Loader for the transport CPU extension."""

from __future__ import annotations

import functools
import os
import warnings
from typing import Final

import torch
from torch.utils.cpp_extension import load

from torchmatch.transport._loader import PACKAGE_DIR, find_prebuilt, force_jit

__all__ = ["load_cpu"]

_EXT_NAME: Final[str] = "torchmatch_transport_cpu"
_CSRC_DIR: Final = PACKAGE_DIR / "transport" / "matrix" / "cpu"


def _jit_build() -> None:
    # WHY -fno-fast-math: the network-simplex pivot selection is
    # numerically sensitive; -ffast-math can break tie-breaking
    # comparisons and produce wrong pivots.
    load(
        name=_EXT_NAME,
        sources=[
            str(_CSRC_DIR / "ops.cpp"),
            str(_CSRC_DIR / "exact_emd_op.cpp"),
            str(_CSRC_DIR / "exact" / "EMD_wrapper.cpp"),
        ],
        extra_include_paths=[str(_CSRC_DIR), str(_CSRC_DIR / "exact")],
        extra_cflags=["-O3", "-std=c++17", "-fno-fast-math"],
        is_python_module=False,
        verbose=False,
    )


def _register_python_ops() -> None:
    # The Sinkhorn-family custom_ops live entirely in Python; importing
    # the modules triggers the @torch.library.custom_op decorators.
    from torchmatch.transport.matrix import (  # noqa: F401, PLC0415
        _log_sinkhorn,
        _sinkhorn_divergence,
        _unbalanced_sinkhorn,
    )


def _register_exact_emd_fake() -> None:
    from torchmatch.transport.matrix._exact_emd import _register_fake  # noqa: PLC0415

    _register_fake()


@functools.cache
def load_cpu() -> None:
    """
    Register ``torch.ops.transport.*`` (CPU backend).

    Always registers the Python-side Sinkhorn custom_ops; additionally
    loads the C++ ``exact_emd`` extension when a prebuilt ``.so`` is
    shipped in the wheel or a C++ toolchain is available for the JIT
    fallback. With ``TORCHMATCH_SKIP_TRANSPORT=1`` at build time and no
    toolchain at install, the extension is absent and EXACT_EMD fails
    with a targeted RuntimeError at call time; the Sinkhorn backends
    keep working. Set ``TORCHMATCH_DEBUG_LOADER=1`` to surface the
    underlying JIT failure when diagnosing a partially-broken extension.
    """
    _register_python_ops()

    if not force_jit() and (path := find_prebuilt("_transport_cpu_impl")) is not None:
        torch.ops.load_library(str(path))
        _register_exact_emd_fake()
        return
    try:
        _jit_build()
    except Exception as exc:
        # No prebuilt, no toolchain (or sources stripped from the wheel
        # with TORCHMATCH_SKIP_TRANSPORT). Leave EXACT_EMD inert; the
        # Python wrapper detects the missing op and raises a clean error.
        # The opt-in warning lets users diagnose partially-broken loads
        # (compiler ran but library registration failed mid-way).
        if os.environ.get("TORCHMATCH_DEBUG_LOADER") == "1":
            warnings.warn(
                f"torchmatch.transport CPU extension JIT load failed: {exc!r}. "
                "EXACT_EMD will be unavailable; Sinkhorn backends still work.",
                stacklevel=2,
            )
        return
    _register_exact_emd_fake()
