"""Execute notebooks and export them as markdown pages for the docs site.

Usage:
    python scripts/nb_render.py                # render all notebooks
    python scripts/nb_render.py --family assignment
    python scripts/nb_render.py --family transport
    python scripts/nb_render.py --no-execute   # convert only (use cached outputs)

The script:
1. Converts each .ipynb
2. Executes the .ipynb with nbconvert (timeout: 600 s per notebook)
3. Converts the executed .ipynb → .md
4. Fixes image paths so figures land in docs/site/public/notebooks/<family>/
5. Writes the .md to docs/site/content/<family_path>/tutorials/

Prerequisites (install via `uv sync --group notebooks`):
    matplotlib, jupyter, nbconvert, jupytext, ipykernel
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

FAMILIES: dict[str, dict] = {
    "assignment": {
        "nb_dir": ROOT / "notebooks" / "assignment",
        "content_dir": ROOT / "docs/site/content/2.assignment/tutorials",
        "public_dir": ROOT / "docs/site/public/notebooks/assignment",
        "notebooks": [
            ("01_the_assignment_problem", "1.fundamentals"),
            ("02_backends_and_batching", "2.backends"),
            ("03_object_tracking", "3.tracking"),
        ],
    },
    "transport": {
        "nb_dir": ROOT / "notebooks" / "transport",
        "content_dir": ROOT / "docs/site/content/3.transport/tutorials",
        "public_dir": ROOT / "docs/site/public/notebooks/transport",
        "notebooks": [
            ("01_optimal_transport", "1.optimal-transport"),
            ("02_sinkhorn_algorithm", "2.sinkhorn"),
            ("03_point_clouds", "3.point-clouds"),
        ],
    },
}


def run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kw)


def render_notebook(
    nb_py: Path,
    md_out: Path,
    public_fig_dir: Path,
    *,
    execute: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nb_ipynb = tmp_path / nb_py.name

        # 1. Copy .ipynb to tmp dir
        shutil.copy2(nb_py, nb_ipynb)

        # 2. nbconvert: execute (optional)
        if execute:
            run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=600",
                    "--output",
                    str(nb_ipynb),
                    str(nb_ipynb),
                ]
            )

        # 3. nbconvert: → markdown
        md_tmp = tmp_path / nb_ipynb.with_suffix(".md").name
        run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                f"--output={md_tmp.name}",
                f"--output-dir={tmp}",
                str(nb_ipynb),
            ]
        )

        # 4. Move figure files into docs/site/public/
        public_fig_dir.mkdir(parents=True, exist_ok=True)
        fig_subdir = tmp_path / (nb_ipynb.stem + "_files")
        if fig_subdir.is_dir():
            for fig in fig_subdir.iterdir():
                shutil.copy2(fig, public_fig_dir / fig.name)

        # 5. Fix image paths in the markdown
        md_text = md_tmp.read_text()
        family = public_fig_dir.parent.name  # "assignment" or "transport"

        # nbconvert writes e.g.
        # `![png](01_the_assignment_problem_files/01a_cost_matrix.png)`
        # We rewrite to `/notebooks/assignment/01a_cost_matrix.png`
        def fix_image_path(m: re.Match) -> str:
            fname = Path(m.group(2)).name
            return f"![{m.group(1)}](/notebooks/{family}/{fname})"

        md_text = re.sub(
            r"!\[([^\]]*)\]\(([^)]+_files/[^)]+)\)", fix_image_path, md_text
        )

        # 6. Add frontmatter if missing
        if not md_text.startswith("---"):
            title = nb_py.stem.replace("_", " ").title()
            md_text = f"---\ntitle: {title}\n---\n\n" + md_text

        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(md_text)
        print(f"  → {md_out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family",
        choices=list(FAMILIES),
        default=None,
        help="Render only this family (default: all)",
    )
    ap.add_argument(
        "--no-execute",
        action="store_true",
        help="Skip execution; convert already-executed .ipynb only",
    )
    args = ap.parse_args()

    families = {args.family: FAMILIES[args.family]} if args.family else FAMILIES

    for family, cfg in families.items():
        print(f"\n=== {family} ===")
        for nb_stem, md_stem in cfg["notebooks"]:
            nb_py = cfg["nb_dir"] / f"{nb_stem}.ipynb"
            md_out = cfg["content_dir"] / f"{md_stem}.md"
            print(f"\n{nb_py.name} → {md_out.name}")
            if not nb_py.exists():
                print(f"  WARNING: {nb_py} not found, skipping")
                continue
            render_notebook(
                nb_py,
                md_out,
                cfg["public_dir"],
                execute=not args.no_execute,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
