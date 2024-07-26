"""torchmatch - assignment and transport solvers for PyTorch."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("torchmatch")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from torchmatch import assignment, transport

__all__ = ["__version__", "assignment", "transport"]
