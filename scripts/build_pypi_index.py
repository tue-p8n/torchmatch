"""Generate per-variant PEP 503 simple-repository index files.

For each ``+<variant>`` local version segment in the supplied wheels
(e.g. ``+cpu``, ``+cu126``, ``+cu128``, ``+cu130``), the script writes
an index at ``<output>/<variant>/torchmatch/index.html`` with hrefs
pointing at the GitHub Release asset URLs.

The script is **additive**: when an existing index file is present it
parses it, merges in the new wheels (deduping by filename, with the
new file's sha256 winning on conflict), and rewrites. The release
workflow then runs once per tag while the index accumulates every
published wheel across all releases.

Usage::

    python scripts/build_pypi_index.py \\
        --tag v2.0.0 \\
        --wheels dist/ \\
        --output docs/simple/ \\
        --repo khwstolle/torchmatch
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
import typing

# Wheel filename: <dist_name>-<version>[+local]-<python>-<abi>-<platform>.whl
# dist_name uses underscores (PEP 427 normalisation).
# We match torchmatch, torchmatch_assignment, and torchmatch_transport.
_WHEEL_RE = re.compile(
    r"^(?P<dist>torchmatch)"
    r"-(?P<version>\d+\.\d+\.\d+(?:\.\w+\d*)*)\+(?P<variant>[A-Za-z0-9_]+)"
    r"-(?P<rest>.+\.whl)$"
)
_HREF_RE = re.compile(
    r'<a\s+href="(?P<url>[^"#]+)#sha256=(?P<sha>[0-9a-fA-F]+)">(?P<name>[^<]+)</a>'
)

# PEP 503: normalise package name (lowercase, run of [-_.] → single dash)
_NORM_RE = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    return _NORM_RE.sub("-", name).lower()


class WheelEntry(typing.TypedDict):
    filename: str
    url: str
    sha256: str


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_existing_index(path: pathlib.Path) -> list[WheelEntry]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [
        {"url": m["url"], "sha256": m["sha"], "filename": m["name"]}
        for m in _HREF_RE.finditer(text)
    ]


def render_index(pkg_name: str, variant: str, wheels: list[WheelEntry]) -> str:
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "  <head>",
        '    <meta name="pypi:repository-version" content="1.0">',
        f"    <title>Links for {pkg_name} ({variant})</title>",
        "  </head>",
        "  <body>",
        f"    <h1>Links for {pkg_name} ({variant})</h1>",
    ]
    for w in sorted(wheels, key=lambda x: x["filename"]):
        lines.append(
            f'    <a href="{w["url"]}#sha256={w["sha256"]}">{w["filename"]}</a><br/>'
        )
    lines += ["  </body>", "</html>", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
    )
    ap.add_argument("--tag", required=True, help="Release tag (e.g. v2.0.0)")
    ap.add_argument(
        "--wheels", type=pathlib.Path, required=True, help="Dir of built wheels"
    )
    ap.add_argument(
        "--output", type=pathlib.Path, required=True, help="docs/simple/ root"
    )
    ap.add_argument(
        "--repo", default="khwstolle/torchmatch", help="GitHub <owner>/<repo>"
    )
    args = ap.parse_args()

    # Keyed by (variant, normalized_pkg_name) → {filename: WheelEntry}
    by_variant_pkg: dict[tuple[str, str], dict[str, WheelEntry]] = {}
    found = 0
    for whl in sorted(args.wheels.glob("*.whl")):
        m = _WHEEL_RE.match(whl.name)
        if not m:
            print(f"skip (no +variant local segment): {whl.name}", file=sys.stderr)
            continue
        variant = m["variant"]
        pkg_name = normalize_name(m["dist"])
        url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{whl.name}"
        entry: WheelEntry = {
            "filename": whl.name,
            "url": url,
            "sha256": sha256_file(whl),
        }
        by_variant_pkg.setdefault((variant, pkg_name), {})[whl.name] = entry
        found += 1

    if found == 0:
        print(
            "no wheels matched torchmatch*-<version>+<variant>-*.whl",
            file=sys.stderr,
        )
        sys.exit(1)

    for (variant, pkg_name), wheel_map in sorted(by_variant_pkg.items()):
        outdir = args.output / variant / pkg_name
        outdir.mkdir(parents=True, exist_ok=True)
        index_path = outdir / "index.html"

        merged: dict[str, WheelEntry] = {
            e["filename"]: e for e in parse_existing_index(index_path)
        }
        merged.update(wheel_map)  # new wheels win on filename collision

        index_path.write_text(render_index(pkg_name, variant, list(merged.values())))
        added = len(wheel_map)
        total = len(merged)
        print(f"{variant}/{pkg_name}: wrote {index_path} (+{added} new, {total} total)")


if __name__ == "__main__":
    main()
