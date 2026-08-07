"""
Shared load helpers for the torchmatch-assignment extension.

Mirrors the structure of the original torchmatch._loader but scoped to the
assignment sub-package: PACKAGE_DIR points one level up so find_prebuilt()
can locate _assignment_cpu_impl*.so / _assignment_cuda_impl*.so in the
torchmatch/ package root, which is where the extension is installed.
"""

from __future__ import annotations

import os
import pathlib
import sys
import typing
from collections.abc import Callable, Iterable

import torch

__all__ = [
    "PACKAGE_DIR",
    "find_prebuilt",
    "force_jit",
    "load_extension_module",
    "register_batch_row_fakes",
    "register_batch_unpacked_fakes",
    "register_row_fakes",
]

# Points to site-packages/torchmatch/ (one level up from torchmatch/assignment/).
PACKAGE_DIR: typing.Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent


def force_jit() -> bool:
    return os.environ.get("TORCHMATCH_FORCE_JIT") == "1"


def find_prebuilt(stem: str) -> pathlib.Path | None:
    """Locate a prebuilt {stem}*.so inside the torchmatch package, if any."""
    for path in PACKAGE_DIR.glob(f"{stem}*.so"):
        return path
    return None


def find_or_download_cuda_prebuilt(stem: str) -> pathlib.Path | None:
    """Locate a cached CUDA prebuilt .so, or try downloading it from GitHub Releases."""
    try:
        from torchmatch import __version__
    except ImportError:
        return None

    if "unknown" in __version__ or "dev" in __version__:
        return None

    # Detect platform - only precompile for linux x86_64 in CI
    import platform

    if platform.system().lower() != "linux":
        return None
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return None

    # Detect CUDA version
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return None

    parts = torch.version.cuda.split(".")
    if len(parts) < 2:
        return None
    major, minor = parts[0], parts[1]
    variant = f"cu{major}{minor}"

    # Verify active variant is one of the built target matrices
    if variant not in {"cu126", "cu128", "cu130", "cu132"}:
        return None

    # Cache directory: ~/.cache/torchmatch/v{version}/
    filename = f"{stem}.linux_x86_64.{variant}.so"
    cache_dir = pathlib.Path.home() / ".cache" / "torchmatch" / f"v{__version__}"
    cache_path = cache_dir / filename

    if cache_path.exists():
        return cache_path

    # Try to fetch from GitHub Release
    url = f"https://github.com/tue-p8n/torchmatch/releases/download/v{__version__}/{filename}"
    import urllib.request

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        req = urllib.request.Request(
            url, headers={"User-Agent": "torchmatch-downloader"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            with temp_path.open("wb") as f:
                f.write(response.read())
        temp_path.replace(cache_path)
        return cache_path
    except Exception as exc:
        is_404 = hasattr(exc, "code") and exc.code == 404
        if not is_404:
            print(
                f"Warning: failed to download prebuilt CUDA binary from {url} ({exc!r}); "
                "falling back to JIT compilation.",
                file=sys.stderr,
            )
        return None


def load_extension_module(
    stem: str,
    jit_build: Callable[[], None],
    register_fakes: Callable[[], None],
) -> None:
    """Run the prebuilt-or-JIT decision, then register FakeTensor kernels."""
    path = None
    if not force_jit():
        path = find_prebuilt(stem)
        if path is None and "cuda" in stem:
            path = find_or_download_cuda_prebuilt(stem)

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


def _register(
    op_names: Iterable[str],
    fake: Callable[..., typing.Any],
    *,
    namespace: str = "assignment",
) -> None:
    for name in op_names:
        torch.library.register_fake(f"{namespace}::{name}")(fake)


def register_row_fakes(
    op_names: Iterable[str], *, namespace: str = "assignment"
) -> None:
    """(cost: 2D) -> (N,) int64 row->col mapping."""

    def _fake(cost: torch.Tensor) -> torch.Tensor:
        torch._check(cost.dim() == 2, lambda: "cost must be 2D")
        return cost.new_empty(cost.size(0), dtype=torch.long)

    _register(op_names, _fake, namespace=namespace)


def register_batch_row_fakes(
    op_names: Iterable[str], *, namespace: str = "assignment"
) -> None:
    """(costs: (B, N, M)) -> (B, N) int64 per-problem row->col mapping."""

    def _fake(costs: torch.Tensor) -> torch.Tensor:
        torch._check(costs.dim() == 3, lambda: "costs must be 3D (B, N, M)")
        return costs.new_empty(costs.size(0), costs.size(1), dtype=torch.long)

    _register(op_names, _fake, namespace=namespace)


def register_batch_unpacked_fakes(
    op_names: Iterable[str], *, namespace: str = "assignment"
) -> None:
    """(costs: (B, N, M)) -> (matches, ur, uc, n_matched) tuple."""

    def _fake(
        costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch._check(costs.dim() == 3, lambda: "costs must be 3D (B, N, M)")
        b, n, m = costs.size(0), costs.size(1), costs.size(2)
        return (
            costs.new_empty(b, n, 2, dtype=torch.long),
            costs.new_empty(b, n, dtype=torch.long),
            costs.new_empty(b, m, dtype=torch.long),
            costs.new_empty(b, dtype=torch.long),
        )

    _register(op_names, _fake, namespace=namespace)
