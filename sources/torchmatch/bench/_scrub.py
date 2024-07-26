"""Remove host-identifying fields from a pytest-benchmark JSON payload."""

from __future__ import annotations

import socket
from typing import Any

# pytest-benchmark fields stripped because they can leak host identity.
# - machine_info.node: hostname
# - machine_info.release: kernel release (often "6.18.24-cachyos" etc., often unique)
# - machine_info.cpu.hardware_raw: some pytest-benchmark versions stash hostname-ish
#   data here
_TOP_DROP = ("commit_info",)
_MI_DROP = ("node", "release")
_CPU_DROP = ("hardware_raw",)


def _redact_str(value: str, tokens: list[str]) -> str:
    for t in tokens:
        if t and t in value:
            value = value.replace(t, "<host>")
    return value


def _redact_in_place(d: Any, tokens: list[str]) -> None:
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = _redact_str(v, tokens)
            else:
                _redact_in_place(v, tokens)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, str):
                d[i] = _redact_str(v, tokens)
            else:
                _redact_in_place(v, tokens)


def scrub_machine_info(
    data: dict,
    *,
    extra_tokens: list[str] | None = None,
) -> dict:
    """
    Drop hostname/kernel fields in-place and redact stray hostname strings.

    Returns the same dict for convenience. Idempotent.
    """
    tokens: list[str] = list(extra_tokens or [])
    mi = data.get("machine_info") or {}
    host = mi.get("node")
    if isinstance(host, str) and host:
        tokens.append(host)
    try:
        local = socket.gethostname()
    except OSError:
        local = ""
    if local and local not in tokens:
        tokens.append(local)

    for k in _TOP_DROP:
        data.pop(k, None)
    for k in _MI_DROP:
        mi.pop(k, None)
    cpu = mi.get("cpu") or {}
    for k in _CPU_DROP:
        cpu.pop(k, None)
    if cpu:
        mi["cpu"] = cpu
    data["machine_info"] = mi

    # Recursively redact any leftover hostname mentions (e.g. in
    # benchmarks[*].extra_info), since plugins occasionally stash it.
    _redact_in_place(data, tokens)
    return data
