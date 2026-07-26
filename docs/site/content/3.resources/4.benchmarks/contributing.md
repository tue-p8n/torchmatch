---
title: Contributing benchmarks
description: Run the torchmatch benchmark suite on your hardware and submit the result via PR.
---

The benchmark page aggregates pytest-benchmark runs contributed by
users. If you have CPU or CUDA hardware not yet represented, running
the suite and opening a PR with the result is a useful contribution.
The contribution process requires no maintainer review beyond a quick sanity-check of the submitted JSON file.

The suite covers three files:

- `benchmark_single.py` — single-problem assignment solvers (Jonker-Volgenant, Munkres, and Lawler variants)
- `benchmark_batched.py` — batched assignment ops
- `benchmark_transport.py` — optimal-transport solvers (Sinkhorn-family backends on CPU/CUDA;
  the `samples.loss` point-cloud solver, GPU only)

## Prerequisites

- A clone of [torchmatch](https://github.com/khwstolle/torchmatch).
- Python 3.13 and `uv` (or any virtualenv tool).
- A working torchmatch install:

```bash
uv sync --extra cu128 --all-groups  # or --extra cpu / cu126 / cu130
```

The `bench` dependency group includes `py-cpuinfo`, which the machine-registration step uses to read your CPU model. Install it with:

```bash
uv sync --extra cu128 --group bench
```

## One-time: register your machine

```bash
uv run python -m torchmatch.bench init-machine
```

This auto-detects your CPU, GPU, and RAM, then prompts for a
`machine_type` (one of `workstation`, `server`, `hpc`, `laptop`,
`mobile`, `edge`) and an optional GitHub handle (displayed next to your results on the benchmark page as credit). It writes
`benchmarks/results/<slug>/machine.json` and prints the slug.

The slug is `{machine_type}_{cpu}_{gpu}`, lowercased with spaces replaced by hyphens — for example:

- `laptop_intel-i7-12700h_nvidia-rtx-a1000`
- `workstation_amd-9950x_nvidia-rtx-4090`
- `server_intel-xeon-platinum-8480_nvidia-h100`
- `laptop_apple-m3-max_cpu-only`

If someone has already submitted results for the same hardware, your new run is stored alongside theirs under the same slug directory; the timestamp in the filename keeps each run separate.

## Per-release: collect a run

```bash
uv run python -m torchmatch.bench collect
```

This runs `tests/benchmark_single.py` + `tests/benchmark_batched.py`
through pytest-benchmark, strips your hostname and OS kernel version string from the JSON, and writes the result to:

```
benchmarks/results/<slug>/torchmatch-<version>+py<X.Y>+torch<X.Y>+<variant>-<utc-timestamp>Z.json
```

`<variant>` is one of `cpu`, `cu126`, `cu128`, `cu130` and matches
your installed PyTorch wheel.

A full sweep takes 10 to 30 minutes depending on the box.

## What gets scrubbed

The CLI strips the following from the pytest-benchmark JSON before
writing:

- `machine_info.node` (hostname).
- `machine_info.release` (kernel release; often unique per host).
- `machine_info.cpu.hardware_raw` (occasionally contains hostname-ish
  data).
- Any other string in the JSON that matches your hostname is rewritten
  to `<host>`.
- `commit_info` block (present only if you ran pytest-benchmark with `--benchmark-save` directly, rather than through the `torchmatch.bench collect` wrapper).

The CPU brand (e.g. `12th Gen Intel(R) Core(TM) i7-12700H`), Python
version, and PyTorch version are kept. They are not personally
identifying and they are necessary for the report.

You should inspect the JSON yourself before opening the PR. The
validation CI run will also re-check that no `node`, `release`,
or `/home/<user>` fragments remain.

## Open the PR

```bash
git checkout -b bench/<slug>-<short-context>
git add benchmarks/results/<slug>/
git commit -m "bench: contribute results for <slug>"
git push -u origin HEAD
gh pr create
```

The PR should touch only files under `benchmarks/results/`. The
benchmark page is rebuilt automatically when the PR is merged, so your results will appear on the site after merge without any further action.

## Reviewer checklist

- [ ] All new files are under `benchmarks/results/<existing-or-new-slug>/`.
- [ ] If a new slug appears, it ships a `machine.json` alongside.
- [ ] Filename matches
  `torchmatch-<X.Y.Z>+py<X.Y>+torch<X.Y>+<variant>-<YYYYMMDDTHHMMSSZ>.json`.
- [ ] `benchmark-validate` GitHub Action is green.
- [ ] No file outside `benchmarks/results/**` is changed in a results PR.

## Re-running on the same machine

Re-running after a torchmatch release simply produces a new
timestamped file under the same slug. Old runs are not deleted; the
aggregator picks all of them up and the charts show every run.
