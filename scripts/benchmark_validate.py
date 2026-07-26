"""Validate per-machine benchmark JSONs and machine.json sidecars.

Used by the contributor PR workflow and by CI to confirm that submitted
files match the schema, do not leak host-identifying data, and live
under the right slug directory.

Usage:
    python scripts/benchmark_validate.py [path ...]

Paths may be individual JSON files, slug directories, or
`benchmarks/results/`. Defaults to `benchmarks/results/`.

Exits 0 on success, 1 on any validation failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

_FILENAME_RE = re.compile(
    r"^torchmatch-(?P<version>[^+]+)"
    r"\+py(?P<py>\d+\.\d+)"
    r"\+torch(?P<torch>\d+\.\d+)"
    r"\+(?P<variant>cpu|cu\d+)"
    r"-(?P<ts>\d{8}T\d{6}Z)\.json$"
)

_PEP440_RE = re.compile(
    r"^\d+(?:\.\d+){0,2}"
    r"(?:[abc]\d+|rc\d+)?"
    r"(?:\.post\d+)?"
    r"(?:\.dev\d+)?"
    r"(?:\+[a-zA-Z0-9.]+)?$"
)

_USER_PATH_RE = re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+")
_LEAK_KEYS = ("node", "release")


def _collect_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _collect_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _collect_strings(v)


def _validate_machine_json(path: Path, errors: list[str]) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid JSON ({exc})")
        return None
    expected_slug = path.parent.name
    slug = data.get("slug")
    if slug != expected_slug:
        errors.append(
            f"{path}: slug field {slug!r} does not match directory {expected_slug!r}"
        )
    if not isinstance(data.get("machine_type"), str):
        errors.append(f"{path}: machine_type missing or not a string")
    return data


def _validate_run_json(path: Path, errors: list[str]) -> None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        errors.append(f"{path}: filename does not match the canonical pattern")
        return
    parts = m.groupdict()
    if not _PEP440_RE.match(parts["version"]):
        errors.append(f"{path}: version {parts['version']!r} is not PEP 440-shaped")

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid JSON ({exc})")
        return

    mi = data.get("machine_info") or {}
    for k in _LEAK_KEYS:
        if k in mi:
            errors.append(f"{path}: machine_info.{k} is present (host leak)")
    cpu = mi.get("cpu") or {}
    if "hardware_raw" in cpu:
        errors.append(f"{path}: machine_info.cpu.hardware_raw is present (host leak)")
    if "commit_info" in data:
        errors.append(f"{path}: commit_info present (potential leak)")

    for s in _collect_strings(data):
        match = _USER_PATH_RE.search(s)
        if match is not None:
            errors.append(
                f"{path}: contains user-home path fragment ({match.group(0)!r})"
            )
            break

    if not isinstance(data.get("benchmarks"), list):
        errors.append(f"{path}: benchmarks field missing or not a list")


def _validate_slug_dir(slug_dir: Path, errors: list[str]) -> None:
    machine_path = slug_dir / "machine.json"
    if not machine_path.is_file():
        errors.append(f"{slug_dir}: machine.json is missing")
        return
    _validate_machine_json(machine_path, errors)
    run_files = sorted(slug_dir.glob("torchmatch-*.json"))
    if not run_files:
        # Empty slug dirs are allowed (a machine registered but never collected).
        return
    for run_path in run_files:
        _validate_run_json(run_path, errors)


def _walk(targets: list[Path], errors: list[str]) -> None:
    for t in targets:
        if t.is_file():
            if t.name == "machine.json":
                _validate_machine_json(t, errors)
            elif t.name.startswith("torchmatch-") and t.suffix == ".json":
                _validate_run_json(t, errors)
            else:
                errors.append(f"{t}: unrecognized file name")
        elif t.is_dir():
            if (t / "machine.json").exists():
                _validate_slug_dir(t, errors)
            else:
                for child in sorted(t.iterdir()):
                    if child.is_dir():
                        _validate_slug_dir(child, errors)
        else:
            errors.append(f"{t}: no such path")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="files or directories to validate (default: benchmarks/results)",
    )
    args = ap.parse_args()
    targets = args.targets or [Path("benchmarks/results")]

    # When the default root is missing entirely there is simply nothing to
    # validate; treat it as success so CI does not fail on a clean checkout.
    if not args.targets and not targets[0].exists():
        print("ok (no benchmarks/results directory)")
        return 0

    errors: list[str] = []
    _walk(targets, errors)

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"\n{len(errors)} validation error(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
