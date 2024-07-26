"""End-to-end benchmark collection: invoke pytest, scrub, write."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from torchmatch.bench._filename import current_run_id
from torchmatch.bench._scrub import scrub_machine_info

_DEFAULT_TEST_FILES = (
    "benchmark_single.py",
    "benchmark_batched.py",
    "benchmark_transport.py",
)


def _resolve_tests(tests_dir: Path) -> list[Path]:
    files = [tests_dir / name for name in _DEFAULT_TEST_FILES]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        msg = (
            "benchmark scripts not found: "
            + ", ".join(missing)
            + "; pass --tests-dir to point at the tests/ directory"
        )
        raise FileNotFoundError(msg)
    return files


def collect_run(
    results_root: Path,
    *,
    slug: str | None = None,
    tests_dir: Path | None = None,
    extra_pytest_args: Sequence[str] = (),
    run_id: str | None = None,
) -> Path:
    """
    Run the benchmark sweep and write a scrubbed JSON to the slug dir.

    ``slug`` defaults to a unique directory under ``results_root``
    containing a ``machine.json`` (auto-detection). ``tests_dir`` defaults
    to ``<cwd>/tests``. Returns the path of the written file.
    """
    if tests_dir is None:
        tests_dir = Path.cwd() / "tests"
    test_files = _resolve_tests(tests_dir)

    if slug is None:
        slug = _autodetect_slug(results_root)
    slug_dir = results_root / slug
    if not (slug_dir / "machine.json").exists():
        msg = (
            f"no machine.json under {slug_dir}; "
            "run `python -m torchmatch.bench init-machine` first"
        )
        raise FileNotFoundError(msg)

    if run_id is None:
        run_id = current_run_id()

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
        dir=tempfile.gettempdir(),
    ) as tf:
        tmp_json = Path(tf.name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *[str(p) for p in test_files],
            "--benchmark-only",
            f"--benchmark-json={tmp_json}",
            *extra_pytest_args,
        ]
        subprocess.run(cmd, check=True)
        data = json.loads(tmp_json.read_text())
    finally:
        tmp_json.unlink(missing_ok=True)

    _drop_raw_samples(data)
    data = scrub_machine_info(data)
    out_path = slug_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    return out_path


def _drop_raw_samples(data: dict) -> None:
    """
    Strip pytest-benchmark's per-iteration ``stats.data`` arrays in place.

    The summary stats (median, mean, min, ...) are retained; the raw sample
    array is ~86% of each run file and nothing downstream consumes it:
    aggregate reads only the summary keys, and validate checks filenames,
    slugs, and host-leak patterns. Dropping the samples before
    ``scrub_machine_info`` also spares the scrubber from walking them.
    """
    for bench in data.get("benchmarks") or []:
        if not isinstance(bench, dict):
            continue
        stats = bench.get("stats")
        if isinstance(stats, dict):
            stats.pop("data", None)


def _autodetect_slug(results_root: Path) -> str:
    """
    Pick the unique slug under results_root containing machine.json.

    Raises if 0 or 2+ candidates exist; in that case the caller must
    pass --slug explicitly.
    """
    if not results_root.exists():
        msg = f"results root {results_root} does not exist"
        raise FileNotFoundError(msg)
    candidates = [p for p in results_root.iterdir() if (p / "machine.json").is_file()]
    if not candidates:
        msg = (
            f"no machine.json found under {results_root}; "
            "run `python -m torchmatch.bench init-machine` first"
        )
        raise FileNotFoundError(msg)
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        msg = (
            f"multiple machine.json files under {results_root} ({names}); "
            "pass --slug to disambiguate"
        )
        raise ValueError(msg)
    return candidates[0].name
