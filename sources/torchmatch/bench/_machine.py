"""Hardware detection and slug construction for the benchmark CLI."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Literal

import torch

try:
    import cpuinfo  # py-cpuinfo (optional dependency in the [bench] group).
except ImportError:
    cpuinfo = None  # type: ignore[assignment]

MachineType = Literal["workstation", "server", "hpc", "laptop", "mobile", "edge"]
_VALID_MACHINE_TYPES: tuple[MachineType, ...] = (
    "workstation",
    "server",
    "hpc",
    "laptop",
    "mobile",
    "edge",
)

SCHEMA_VERSION = 1


def short_name(brand: str) -> str:
    """
    Normalize a noisy CPU/GPU brand string to a filesystem-safe short slug.

    Drops marketing tokens ("(R)", "(TM)", "Core", "Processor", "Laptop GPU",
    "GeForce"...) and runs of non-alphanumerics. Lowercased.

    Examples:
        '12th Gen Intel(R) Core(TM) i7-12700H' -> 'intel-i7-12700h'
        'AMD Ryzen 9 9950X 16-Core Processor'  -> 'amd-ryzen-9-9950x'
        'NVIDIA RTX A1000 Laptop GPU'          -> 'nvidia-rtx-a1000'

    """
    s = brand.lower()
    s = re.sub(r"\((?:r|tm|c)\)", "", s)
    s = re.sub(r"\b\d+th gen\b", "", s)
    s = re.sub(r"\b(?:core|processor|cpu|gpu|laptop|geforce)\b", "", s)
    s = re.sub(r"\b\d+-core\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _read_proc_cpuinfo() -> dict[str, str]:
    """Pull the first 'model name' / sibling block from /proc/cpuinfo."""
    info: dict[str, str] = {}
    p = Path("/proc/cpuinfo")
    if not p.exists():
        return info
    for line in p.read_text().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        if k and k not in info:
            info[k] = v
    return info


def _physical_cores_linux() -> int | None:
    # Count distinct (physical id, core id) pairs in /proc/cpuinfo.
    p = Path("/proc/cpuinfo")
    if not p.exists():
        return None
    pairs: set[tuple[str, str]] = set()
    cur: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            if "physical id" in cur and "core id" in cur:
                pairs.add((cur["physical id"], cur["core id"]))
            cur = {}
            continue
        k, _, v = line.partition(":")
        cur[k.strip()] = v.strip()
    if "physical id" in cur and "core id" in cur:
        pairs.add((cur["physical id"], cur["core id"]))
    return len(pairs) or None


def detect_cpu() -> dict:
    """Return a dict with brand / short / physical_cores / logical_cores."""
    brand = ""
    if cpuinfo is not None:
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw") or ""
    if not brand:
        proc_info = _read_proc_cpuinfo()
        brand = proc_info.get("model name") or platform.processor() or "unknown"
    logical = os.cpu_count() or 0
    physical = _physical_cores_linux() or logical
    return {
        "brand": brand,
        "short": short_name(brand) or "unknown-cpu",
        "physical_cores": physical,
        "logical_cores": logical,
    }


def detect_gpu() -> dict | None:
    """Return a dict for the first CUDA device, or None on CPU-only builds."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    return {
        "brand": name,
        "short": short_name(name) or "unknown-gpu",
        "compute_capability": cc,
    }


def detect_ram() -> dict:
    """Return total system memory (GB) from /proc/meminfo when available."""
    p = Path("/proc/meminfo")
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return {"total_gb": round(kb / 1024 / 1024)}
    return {}


def detect_os() -> dict:
    return {"name": platform.system()}


def build_slug(
    machine_type: MachineType,
    cpu_short: str,
    gpu_short: str | None,
) -> str:
    """Compose the flat slug used as the per-machine directory name."""
    gpu = gpu_short or "cpu-only"
    return f"{machine_type}_{cpu_short}_{gpu}"


def detect_machine(machine_type: MachineType) -> dict:
    """Full machine.json payload for the local box."""
    if machine_type not in _VALID_MACHINE_TYPES:
        msg = (
            f"machine_type must be one of {_VALID_MACHINE_TYPES!r}, "
            f"got {machine_type!r}"
        )
        raise ValueError(msg)
    cpu = detect_cpu()
    gpu = detect_gpu()
    slug = build_slug(machine_type, cpu["short"], gpu["short"] if gpu else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "slug": slug,
        "machine_type": machine_type,
        "cpu": cpu,
        "gpu": gpu,
        "ram": detect_ram(),
        "os": detect_os(),
        "submitted_by": "",
        "notes": "",
    }


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{text}{suffix}: ").strip()
    return raw or default


def _prompt_machine_type() -> MachineType:
    types = "/".join(_VALID_MACHINE_TYPES)
    while True:
        val = _prompt(f"Machine type ({types})")
        if val in _VALID_MACHINE_TYPES:
            return val  # type: ignore[return-value]
        print(f"  not one of {types}; try again")  # noqa: T201


def init_machine(  # noqa: PLR0913
    results_root: Path,
    *,
    machine_type: MachineType | None = None,
    submitted_by: str = "",
    notes: str = "",
    interactive: bool = True,
    force: bool = False,
) -> Path:
    """
    Write benchmarks/results/<slug>/machine.json and return its path.

    If a machine.json already exists at the derived slug, refuses to
    overwrite unless ``force=True``.
    """
    if machine_type is None:
        if not interactive:
            msg = "machine_type required when interactive=False"
            raise ValueError(msg)
        machine_type = _prompt_machine_type()
    payload = detect_machine(machine_type)
    if interactive:
        payload["submitted_by"] = _prompt("GitHub handle (optional)", submitted_by)
        payload["notes"] = _prompt("Notes (optional)", notes)
    else:
        payload["submitted_by"] = submitted_by
        payload["notes"] = notes
    slug_dir = results_root / payload["slug"]
    target = slug_dir / "machine.json"
    if target.exists() and not force:
        msg = f"machine.json already exists at {target}; pass --force to overwrite"
        raise FileExistsError(msg)
    slug_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def detect_driver() -> str | None:
    """NVIDIA driver version via nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    return out.strip().splitlines()[0] if out.strip() else None
