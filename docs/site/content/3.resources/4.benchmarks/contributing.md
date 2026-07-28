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

Two ways to run it: with Nix (below), or without Nix at all via a
prebuilt container image — see [Without Nix](#without-nix-apptainer-hpc),
useful on HPC login/compute nodes where installing Nix isn't practical.

## Prerequisites

- A clone of [torchmatch](https://github.com/tue-p8n/torchmatch).
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

## Without Nix (Apptainer / HPC)

On a shared HPC login/compute node where installing Nix isn't practical,
pull the prebuilt image and run it with [Apptainer](https://apptainer.org)
instead — no Nix, no root, no daemon. The image is rebuilt from `main`
on every push that touches the solver sources or tests (see
`.github/workflows/bench-image.yml`), so pull again before a new run to
pick up changes.

`--pwd /app` is required on every invocation below: Apptainer preserves
your shell's own working directory by default, which silently writes
results outside the bind mount instead of into it.

One-time machine registration:

```bash
mkdir -p benchmarks/results
apptainer run --nv --pwd /app \
  --bind "$PWD/benchmarks/results:/app/benchmarks/results" \
  docker://ghcr.io/tue-p8n/torchmatch/bench:latest \
  init-machine
```

Per-release collection:

```bash
apptainer run --nv --pwd /app \
  --bind "$PWD/benchmarks/results:/app/benchmarks/results" \
  docker://ghcr.io/tue-p8n/torchmatch/bench:latest \
  collect
```

Same CLI, same output layout, same scrubbing, same 10–30 minute
runtime as the Nix path above — `--nv` maps the host's NVIDIA driver
into the container at runtime, so no CUDA toolkit install is needed on
the host itself. Docker with `--gpus all` works too, provided the
NVIDIA Container Toolkit is set up on that host (Apptainer's `--nv`
needs no such extra toolkit — it's the more portable option on a
cluster you don't administer).

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

## Commit and push

```bash
git add benchmarks/results/<slug>/
git commit -m "bench: contribute results for <slug>"
git fetch origin main
git push origin main
```

Push directly to `main` — this repo doesn't use PRs for contributions.
Commit only files under `benchmarks/results/`. The benchmark page is
rebuilt automatically on the next docs deploy, and `benchmark-validate`
runs on the push itself to catch scrubbing/format mistakes.

Before pushing, double check:

- [ ] All new files are under `benchmarks/results/<existing-or-new-slug>/`.
- [ ] If a new slug appears, it ships a `machine.json` alongside.
- [ ] Filename matches
  `torchmatch-<X.Y.Z>+py<X.Y>+torch<X.Y>+<variant>-<YYYYMMDDTHHMMSSZ>.json`.
- [ ] No file outside `benchmarks/results/**` is staged.

Or just run `nix run .#bench-validate` (or, without Nix, `python
scripts/benchmark_validate.py benchmarks/results`) before committing —
the same check CI runs.

## Re-running on the same machine

Re-running after a torchmatch release simply produces a new
timestamped file under the same slug. Old runs are not deleted; the
aggregator picks all of them up and the charts show every run.
