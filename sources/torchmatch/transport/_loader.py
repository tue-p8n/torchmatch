"""
Shared load helpers for the torchmatch-transport extension.

Mirrors the structure of the original torchmatch._loader but scoped to the
transport sub-package: PACKAGE_DIR points one level up so find_prebuilt()
can locate _transport_cpu_impl*.so / _transport_cuda_impl*.so in the
torchmatch/ package root.
"""

from __future__ import annotations

import functools
import os
import pathlib
import sys
import typing
from collections.abc import Callable, Iterable

import torch

__all__ = [
    "PACKAGE_DIR",
    "ensure_transport_namespace",
    "find_prebuilt",
    "force_jit",
    "load_extension_module",
    "register_batch_plan_fakes",
    "register_divergence_fakes",
]

# Points to site-packages/torchmatch/ (one level up from torchmatch/transport/).
PACKAGE_DIR: typing.Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent


def force_jit() -> bool:
    return os.environ.get("TORCHMATCH_FORCE_JIT") == "1"


def find_prebuilt(stem: str) -> pathlib.Path | None:
    """Locate a prebuilt {stem}*.so inside the torchmatch package, if any."""
    for path in PACKAGE_DIR.glob(f"{stem}*.so"):
        return path
    return None


def load_extension_module(
    stem: str,
    jit_build: Callable[[], None],
    register_fakes: Callable[[], None],
) -> None:
    """Run the prebuilt-or-JIT decision, then register FakeTensor kernels."""
    path = None if force_jit() else find_prebuilt(stem)
    if path is not None:
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:
            print(
                f"Warning: failed to load prebuilt {path.name} ({exc}); "
                "falling back to JIT compilation.",
                file=sys.stderr,
            )
            jit_build()
    else:
        jit_build()
    register_fakes()


@functools.cache
def _transport_library() -> torch.library.Library:
    return torch.library.Library("transport", "FRAGMENT")


def ensure_transport_namespace() -> None:
    """
    Register an empty torch.ops.transport fragment from Python.

    Lets the transport sub-package publish its namespace without forcing a
    C++ toolchain. Idempotent; safe to call from both the CPU and CUDA loaders.
    """
    _transport_library()


def _register(
    op_names: Iterable[str],
    fake: Callable[..., typing.Any],
    *,
    namespace: str = "transport",
) -> None:
    for name in op_names:
        torch.library.register_fake(f"{namespace}::{name}")(fake)


def register_batch_plan_fakes(
    op_names: Iterable[str], *, namespace: str = "transport"
) -> None:
    """(costs: (B, N, M)) -> (B, N, M) transport plan, same dtype as costs."""

    def _fake(costs: torch.Tensor) -> torch.Tensor:
        torch._check(costs.dim() == 3, lambda: "costs must be 3D (B, N, M)")
        return costs.new_empty(costs.size(0), costs.size(1), costs.size(2))

    _register(op_names, _fake, namespace=namespace)


def register_divergence_fakes(
    op_names: Iterable[str], *, namespace: str = "transport"
) -> None:
    """(costs: (B, N, M)) -> (B,) scalar divergence per batch."""

    def _fake(costs: torch.Tensor) -> torch.Tensor:
        torch._check(costs.dim() == 3, lambda: "costs must be 3D (B, N, M)")
        return costs.new_empty(costs.size(0))

    _register(op_names, _fake, namespace=namespace)
