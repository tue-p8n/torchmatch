"""Aggregate per-machine benchmark JSONs into the datasets the docs site consumes.

Walks ``benchmarks/results/<slug>/`` for every run file, flattens each
benchmark into one row, and emits:

- ``docs/site/public/benchmarks.json``: ``[{slug, run_id, version, py_version,
  torch_version, variant, ts, group, op, n, b, dtype, dist, stats}]``
- ``docs/site/public/machines.json``: ``[{slug, machine_type, cpu_brand,
  gpu_brand, ram_gb, run_count, latest_run, raw}]``

Stdlib-only so the docs CI can run it without installing torchmatch.

Usage:

    python scripts/benchmark_aggregate.py
    python scripts/benchmark_aggregate.py --results-root benchmarks/results \\
        --out-dir docs/site/public
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

_DTYPE_ALIASES = {
    "UNSERIALIZABLE[torch.float32]": "f32",
    "UNSERIALIZABLE[torch.float64]": "f64",
    "torch.float32": "f32",
    "torch.float64": "f64",
}


def _parse_filename(name: str) -> dict | None:
    m = _FILENAME_RE.match(name)
    return m.groupdict() if m else None


def _normalize_dtype(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw)
    return _DTYPE_ALIASES.get(s, s)


def _stat(stats: dict, key: str) -> float | int | None:
    v = stats.get(key)
    if isinstance(v, (int, float)):
        return v
    return None


def _flatten_run(
    slug: str,
    run_id: str,
    meta: dict,
    data: dict,
) -> Iterable[dict]:
    for b in data.get("benchmarks") or []:
        params = dict(b.get("params") or {})
        stats = b.get("stats") or {}
        yield {
            "slug": slug,
            "run_id": run_id,
            "version": meta["version"],
            "py_version": meta["py"],
            "torch_version": meta["torch"],
            "variant": meta["variant"],
            "ts": meta["ts"],
            "group": b.get("group"),
            "op": params.get("op_name"),
            "n": params.get("n"),
            "b": params.get("b"),
            "dim": params.get("dim"),
            "dtype": _normalize_dtype(params.get("dtype")),
            "dist": params.get("dist"),
            "stats": {
                "median": _stat(stats, "median"),
                "mean": _stat(stats, "mean"),
                "stddev": _stat(stats, "stddev"),
                "min": _stat(stats, "min"),
                "max": _stat(stats, "max"),
                "rounds": _stat(stats, "rounds"),
                "iterations": _stat(stats, "iterations"),
            },
        }


def _machine_summary(slug: str, machine: dict, runs: list[dict]) -> dict:
    cpu = machine.get("cpu") or {}
    gpu = machine.get("gpu") or {}
    ram = machine.get("ram") or {}
    timestamps = sorted(r["ts"] for r in runs)
    return {
        "slug": slug,
        "machine_type": machine.get("machine_type"),
        "cpu_brand": cpu.get("brand"),
        "gpu_brand": gpu.get("brand"),
        "ram_gb": ram.get("total_gb"),
        "run_count": len(runs),
        "latest_run": timestamps[-1] if timestamps else None,
        "raw": machine,
    }


def aggregate(results_root: Path) -> tuple[list[dict], list[dict]]:
    """Walk results_root and return (samples, machines) lists."""
    samples: list[dict] = []
    machines: list[dict] = []
    if not results_root.exists():
        return samples, machines
    for slug_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        machine_path = slug_dir / "machine.json"
        if not machine_path.is_file():
            continue
        try:
            machine = json.loads(machine_path.read_text())
        except json.JSONDecodeError as exc:
            msg = f"failed to parse {machine_path}: {exc}"
            raise RuntimeError(msg) from exc
        slug = machine.get("slug") or slug_dir.name
        run_metas: list[dict] = []
        for run_path in sorted(slug_dir.glob("torchmatch-*.json")):
            meta = _parse_filename(run_path.name)
            if meta is None:
                print(
                    f"skip {run_path.name}: filename does not match the schema",
                    file=sys.stderr,
                )
                continue
            try:
                data = json.loads(run_path.read_text())
            except json.JSONDecodeError as exc:
                msg = f"failed to parse {run_path}: {exc}"
                raise RuntimeError(msg) from exc
            run_id = run_path.stem
            run_metas.append({"run_id": run_id, **meta})
            samples.extend(_flatten_run(slug, run_id, meta, data))
        machines.append(_machine_summary(slug, machine, run_metas))
    return samples, machines


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
        help="root of the per-machine results tree (default: %(default)s)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/site/public"),
        help="where to write benchmarks.json + machines.json (default: %(default)s)",
    )
    args = ap.parse_args()

    samples, machines = aggregate(args.results_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = args.out_dir / "benchmarks.json"
    mach_path = args.out_dir / "machines.json"
    bench_path.write_text(json.dumps(samples, separators=(",", ":")))
    mach_path.write_text(json.dumps(machines, indent=2) + "\n")
    print(
        f"wrote {bench_path} ({len(samples)} samples) "
        f"and {mach_path} ({len(machines)} machines)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
