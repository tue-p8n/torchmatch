"""Canonical filename for a benchmark run."""

from __future__ import annotations

import datetime as _dt
import re
import sys

import torch

import torchmatch

_FILENAME_RE = re.compile(
    r"^torchmatch-(?P<version>[^+]+)"
    r"\+py(?P<py>\d+\.\d+)"
    r"\+torch(?P<torch>\d+\.\d+)"
    r"\+(?P<variant>cpu|cu\d+)"
    r"-(?P<ts>\d{8}T\d{6}Z)\.json$"
)


def detect_variant() -> str:
    """
    Map the running PyTorch build to a wheel-variant tag.

    Returns 'cpu' for CPU-only builds; 'cuXYZ' otherwise (matches the
    wheel suffixes produced by release.yml).
    """
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return "cpu"
    # torch.version.cuda looks like '12.8' or '13.0'. Wheel suffix is
    # 'cu128' / 'cu130' (concatenated, dropping the dot).
    return "cu" + cuda.replace(".", "")


def _xy(triple: str) -> str:
    """Reduce 'X.Y.Z'-ish version strings to 'X.Y'."""
    parts = triple.split(".")
    if len(parts) < 2:
        return triple
    return f"{parts[0]}.{parts[1]}"


def py_version_xy() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def torch_version_xy() -> str:
    return _xy(torch.__version__.split("+", 1)[0])


def torchmatch_version() -> str:
    """Return the full PEP 440 version of the installed torchmatch."""
    return torchmatch.__version__


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def make_filename(
    *,
    version: str,
    py: str,
    torch_ver: str,
    variant: str,
    ts: str,
) -> str:
    return f"torchmatch-{version}+py{py}+torch{torch_ver}+{variant}-{ts}.json"


def current_run_id() -> str:
    """Filename (sans .json) for a run captured right now on this interpreter."""
    name = make_filename(
        version=torchmatch_version(),
        py=py_version_xy(),
        torch_ver=torch_version_xy(),
        variant=detect_variant(),
        ts=_utc_ts(),
    )
    return name.removesuffix(".json")


def parse_filename(name: str) -> dict | None:
    """Reverse a canonical filename into its component dict, or None."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.groupdict()
