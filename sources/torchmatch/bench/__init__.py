"""
Benchmark collection CLI.

Helpers to capture a pytest-benchmark sweep on the local machine,
scrub host-identifying fields, and write the result under
``benchmarks/results/<slug>/`` with a canonical filename that
encodes the torchmatch / Python / PyTorch versions and the CUDA
wheel variant.

The full contributor flow is :func:`init_machine` (once) +
:func:`collect_run` (once per release). The aggregator under
``scripts/benchmark_aggregate.py`` then turns the per-machine
files into the JSON datasets that the docs site consumes.
"""

from __future__ import annotations

from torchmatch.bench._collect import collect_run
from torchmatch.bench._filename import (
    current_run_id,
    detect_variant,
    make_filename,
    parse_filename,
)
from torchmatch.bench._machine import (
    build_slug,
    detect_machine,
    init_machine,
    short_name,
)
from torchmatch.bench._scrub import scrub_machine_info

__all__ = [
    "build_slug",
    "collect_run",
    "current_run_id",
    "detect_machine",
    "detect_variant",
    "init_machine",
    "make_filename",
    "parse_filename",
    "scrub_machine_info",
    "short_name",
]
