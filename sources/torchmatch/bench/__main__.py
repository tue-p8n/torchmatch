"""``python -m torchmatch.bench`` CLI dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchmatch.bench._collect import collect_run
from torchmatch.bench._machine import _VALID_MACHINE_TYPES, init_machine
from torchmatch.bench._scrub import scrub_machine_info

_DEFAULT_RESULTS = Path("benchmarks") / "results"


def _cmd_init_machine(args: argparse.Namespace) -> int:
    path = init_machine(
        args.results_root,
        machine_type=args.type,
        submitted_by=args.submitted_by or "",
        notes=args.notes or "",
        interactive=args.type is None,
        force=args.force,
    )
    payload = json.loads(path.read_text())
    print(f"wrote {path}")  # noqa: T201
    print(f"slug: {payload['slug']}")  # noqa: T201
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    out = collect_run(
        args.results_root,
        slug=args.slug,
        tests_dir=args.tests_dir,
        extra_pytest_args=args.pytest_args,
    )
    print(f"wrote {out}")  # noqa: T201
    return 0


def _cmd_scrub(args: argparse.Namespace) -> int:
    data = json.loads(args.input.read_text())
    data = scrub_machine_info(data)
    out = args.output or args.input
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {out}")  # noqa: T201
    return 0


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m torchmatch.bench",
        description="Capture, scrub and stage torchmatch benchmark runs.",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=_DEFAULT_RESULTS,
        help="benchmarks/results/ root (default: %(default)s relative to cwd)",
    )
    subs = p.add_subparsers(dest="cmd", required=True)

    init = subs.add_parser("init-machine", help="write machine.json for this host")
    init.add_argument(
        "--type",
        choices=_VALID_MACHINE_TYPES,
        default=None,
        help="machine class; prompts interactively when omitted",
    )
    init.add_argument("--submitted-by", default=None, help="GitHub handle (optional)")
    init.add_argument("--notes", default=None, help="free-form notes")
    init.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing machine.json",
    )
    init.set_defaults(func=_cmd_init_machine)

    coll = subs.add_parser("collect", help="run benchmarks and write a scrubbed JSON")
    coll.add_argument(
        "--slug",
        default=None,
        help="slug under results-root (auto-detected from a unique machine.json)",
    )
    coll.add_argument(
        "--tests-dir",
        type=Path,
        default=None,
        help="path to tests/ (default: <cwd>/tests)",
    )
    coll.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra args forwarded to pytest after --",
    )
    coll.set_defaults(func=_cmd_collect)

    scr = subs.add_parser("scrub", help="strip host-identifying fields from a JSON")
    scr.add_argument("input", type=Path, help="pytest-benchmark JSON to scrub")
    scr.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: overwrite input)",
    )
    scr.set_defaults(func=_cmd_scrub)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m torchmatch.bench``."""
    parser = _make_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
