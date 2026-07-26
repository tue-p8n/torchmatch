# CLAUDE.md

## Project

Single PyPI distribution `torchmatch` containing assignment and transport solvers.

Install with `pip install torchmatch`.

**`torchmatch.assignment`**: integer LAP solvers. Cost matrix in, int64 row→col out.

- _Shortest-augmenting-path_ (JV lineage): `jonker_scalar`, `jonker_dense`,
  `jonker_compact` (CPU); `jonker_*_batch`, `jonker_*_batch_unpacked`
  (batched); `jonker_dense_batch` (CPU + CUDA tiled kernel).
- _Primed/starred-zeros_ (Kuhn-Munkres lineage; CUDA only):
  `munkres`, `lawler`, `hybrid`.
- _Heuristic_: `greedy` (single-pass Kurtzberg 1962; pure PyTorch).

**`torchmatch.transport`**: continuous OT solvers.

- `torchmatch.transport.matrix`: cost-matrix face. 2-D or 3-D cost in,
  log-plan or scalar divergence out. Backends: `log_sinkhorn`,
  `sinkhorn_divergence`, `unbalanced_sinkhorn` (Python custom_ops, LSE
  loops, CPU + CUDA); `exact_emd` (network simplex, CPU-only).
- `torchmatch.transport.samples`: point-cloud face. Triton streaming
  kernels (CUDA-only). Entry point: `samples.loss(...)`. Gradient via
  analytic online backward (IFT / CG implicit differentiation path).

Python 3.13 only (`>=3.13, <3.14`). `import torchmatch` lazily re-exports
whichever sub-packages are installed; importing a sub-package directly is
always safe.

## Public API

Per-family pattern: `solve()` (unified dispatcher) and `ops` (direct
`torch.ops.<family>.*` handles).

### `torchmatch.assignment`

- **`solve(cost, *, backend, unpack)`**: dispatcher. `cost` is (N,M) or
  (B,N,M) float32/float64. Returns `int64` row→col tensor, or a
  `(matches, unmatched_rows, unmatched_cols)` tuple if `unpack=True`.
  Unmatched rows return `-1`.
- **`resolve_backend(cost, *, backend)`**: returns the op name AUTO would
  pick — useful for test assertions and debugging.
- **`assignment_cost(cost, matches, *, reduction)`**: total cost of a LAP
  solution. `reduction` ∈ `{"sum", "mean", "none"}`.
- **`Backend`**: `StrEnum` — `AUTO`, `JONKER`, `MUNKRES`, `LAWLER`,
  `GREEDY`.
- **`ops.<op>`**: direct handles — `jonker_scalar`, `jonker_dense`,
  `jonker_compact`, `jonker_dense_batch`, `jonker_compact_batch`,
  `jonker_dense_batch_unpacked`, `jonker_compact_batch_unpacked`, `greedy`;
  `munkres`, `hybrid`, `lawler` when CUDA is available.

### `torchmatch.transport.matrix`

- **`solve(cost, *, backend, reg, n_iter, mask, a, b, scaling, rho, cost_aa, cost_bb, unpack)`**:
  dispatcher. `cost` is (N,M) or (B,N,M). Returns log-plan or scalar
  divergence. `unpack=True` returns `(log_plan, f, g)` dual potentials
  (`f`/`g` are `None` for `EXACT_EMD`).
- **`marginal_error(log_plan, a, b)`**: max absolute deviation of plan
  row/col sums from marginals — useful for checking Sinkhorn convergence.
- **`Backend`**: `StrEnum` — `AUTO`, `LOG_SINKHORN` (AUTO default),
  `SINKHORN_DIVERGENCE`, `UNBALANCED_SINKHORN`, `EXACT_EMD`.
- **`ops.<op>`**: `log_sinkhorn`, `sinkhorn_divergence`,
  `unbalanced_sinkhorn`, `exact_emd`.

### `torchmatch.transport.samples`

- **`loss(x, y, *, blur, debias, reach, reach_x, reach_y, a, b, scaling, threshold, half_cost, p)`**:
  point-cloud OT cost. `x (n,d)`, `y (m,d)`, float32, CUDA. `eps = blur²`.
  `debias=True` computes the Sinkhorn divergence (3 solves). `reach`/
  `reach_x`/`reach_y` → KL marginal penalty `rho = reach²`. `n_iter` is
  intentionally unsupported (symmetric solver runs an eps schedule; tune
  via `scaling`). Batched 3-D input `(B,N,D)` loops over the batch dim.

### Input semantics

- `NaN` and `-inf`: **rejected** at the entry point (signal upstream bugs).
- `+inf`: **forbidden edge**, rewritten to a per-call finite sentinel.

## Dev environment

Every project task is a flake app. `nix run .#<task>` runs it directly;
`nix flake show` lists everything. The five Python-touching apps
(`bench-init`, `bench-collect`, `test`, `lint`, `format`) default to the
`cu128` venv. Per-variant suffixed forms (`test-<variant>`, etc.) pin a
specific ABI.

```bash
nix develop                           # default = cu128 devshell
nix develop .#cpu                     # CPU-only (sets TORCHMATCH_SKIP_CUDA=1)
nix develop .#cu126 / .#cu128 / .#cu130 / .#cu132
uv sync --extra cu128 --all-groups    # editable install
```

`cu128` is frozen at torch 2.11; `cu130`/`cu132` need torch 2.12+. Use `cu126`
for broadest 12.x driver coverage.

## Common commands

```bash
# Tests (timeout=10s per test; conftest imports torchmatch eagerly)
nix run .#test
nix run .#test -- tests/test_lap_jv_scalar.py::test_square_optimal
nix run .#test -- tests/test_lap_opcheck.py     # schema/FakeTensor drift
nix run .#test -- tests/test_transport_smoke.py  # transport load smoke

# Benchmarks (timeout disabled; pytest-benchmark)
nix run .#bench-init       # one-time: write benchmarks/results/<slug>/machine.json
nix run .#bench-collect    # run both benchmark files, write JSON under <slug>/
nix run .#bench-aggregate  # flatten results -> docs/site/public/{benchmarks,machines}.json
nix run .#bench-validate   # schema/leak checks (same as PR CI)

# Build wheel
# Default: JIT-only sdist/wheel (py3-none-any); C++ sources shipped, compiled at first import.
uv build --wheel                                     # JIT wheel — no prebuilt binaries
# Stable ABI CPU wheel (cp313-abi3): works with any torch ≥ 2.10, Python ≥ 3.13.
TORCHMATCH_BUILD_CPU=1 TORCHMATCH_BUILD_TRANSPORT=1 TORCHMATCH_BUILD_STABLE_ABI=1 \
  uv build --wheel                                   # PyPI wheel (stable ABI)
# Pre-compiled wheels without stable ABI (require BUILD_* opt-in):
TORCHMATCH_BUILD_CPU=1 TORCHMATCH_BUILD_TRANSPORT=1 uv build --wheel               # CPU +abi
TORCHMATCH_BUILD_CPU=1 TORCHMATCH_BUILD_CUDA=1 TORCHMATCH_BUILD_TRANSPORT=1 \
  CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" uv build --wheel  # CUDA

# Build via Nix (.so lands in result/)
nix build .#torchmatch-cpu
nix build .#torchmatch-cu128
nix build .#torchmatch-cu130
nix build .#torchmatch-cu132

# Lint and format
nix run .#lint
nix run .#format

# Docs (Nuxt 4 under docs/site/)
nix run .#docs-serve    # live-reload at 127.0.0.1:3000
nix run .#docs-build
nix run .#docs-preview
```

## Repository layout

```
torchmatch/
├── pyproject.toml          ← single package definition
├── setup.py                ← builds all four C++ extensions
└── sources/torchmatch/
    ├── __init__.py         ← imports assignment and transport directly
    ├── bench/              ← torchmatch.bench
    ├── assignment/
    │   ├── _loader.py      ← load helpers (PACKAGE_DIR=torchmatch/)
    │   ├── _cpu.py / _cuda.py
    │   ├── cpu/ cuda/      ← C++/CUDA sources (also shipped for JIT)
    │   └── ...
    └── transport/
        ├── _loader.py      ← load helpers (PACKAGE_DIR=torchmatch/)
        ├── matrix/         ← Sinkhorn + exact EMD
        └── samples/        ← Triton point-cloud solvers
```

`uv sync --extra cu128 --all-groups` at the root installs the package in editable mode.

## Architecture

### Extension loading

Each sub-module's `__init__.py` calls `load_cpu()`, then `load_cuda()` iff
`torch.cuda.is_available()`.

Each sub-module owns its own `_loader.py`:

- `torchmatch/assignment/_loader.py`: assignment load helpers.
- `torchmatch/transport/_loader.py`: transport load helpers.

Both define `PACKAGE_DIR = Path(__file__).resolve().parent.parent` — this
points to `site-packages/torchmatch/`, where the `.so` files land regardless
of which sub-package built them (extension names are `torchmatch._assignment_cpu_impl` etc.).

The **prebuilt-or-JIT** recipe (in each `_loader.py`):

1. `find_prebuilt("_assignment_cpu_impl"|...)` → `torch.ops.load_library`.
2. Otherwise, `torch.utils.cpp_extension.load(...)` JIT-compiles from
   sources shipped inside the package (at `PACKAGE_DIR / "assignment" / "cpu"` etc.).
3. Register **FakeTensor kernels** for `torch.compile` / `torch.export`.

`TORCHMATCH_FORCE_JIT=1` skips step 1.

FakeTensor shapes (`assignment/_loader.py`):

- `register_row_fakes`: `(N,M) → (N,) int64`
- `register_batch_row_fakes`: `(B,N,M) → (B,N) int64`
- `register_batch_unpacked_fakes`: `(B,N,M) → (matches, ur, uc, n_matched)`

FakeTensor shapes (`transport/_loader.py`):

- `register_batch_plan_fakes`: `(B,N,M) → (B,N,M)`
- `register_divergence_fakes`: `(B,N,M) → (B,)`

### Transport sub-package (`sources/torchmatch/transport/`)

- `matrix/`: Python custom_ops for log_sinkhorn / sinkhorn_divergence /
  unbalanced_sinkhorn (pure-Python LSE loops, CPU + CUDA). C++ network
  simplex under `matrix/cpu/exact/` for `exact_emd`. Dispatcher in
  `_solve.py`. `_schedule.py` builds the Schmitzer 2019 eps-scaling
  schedule. `_validate.py` handles input validation and mask fusion.
- `samples/`: Triton streaming kernels under `kernels/`; autograd via
  `torch.library.register_autograd` in `_autograd.py` (analytic online
  backward); implicit gradient via IFT + CG in `_implicit_grad.py` and
  `_cg.py`; `_loss.py` is the user entry point. CUDA-only.

### Op registration (C++/CUDA)

- `assignment/cpu/ops.cpp`: defines JV ops. CPU default; JV ops accept
  rectangular (N,M) and pad to square internally.
- `assignment/cuda/ops.cpp`: `TORCH_LIBRARY_FRAGMENT` adds `munkres`,
  `hybrid`, `lawler`, tagged `cudagraph_unsafe` (host-side
  `cudaStreamSynchronize` on managed memory).
- `assignment/cuda/dispatch.cu`: CUDA impls for the three primed-zeros ops
  and `jonker_dense_batch` (single-block-per-problem shared-memory, fully
  capturable; square only, K ≤ 64).
- Each primed-zeros `.cu` is its own translation unit (shared `__managed__`
  / `__constant__` symbol names would collide).
- `transport/matrix/cpu/ops.cpp` + `exact_emd_op.cpp`: declares and
  implements `exact_emd`. No CUDA EMD.
- Sinkhorn-family ops and `_sinkhorn_samples_fwd` are Python
  `@torch.library.custom_op` registrations.

Without `TORCHMATCH_BUILD_TRANSPORT=1`, the transport CPU extension is
absent; Python Sinkhorn ops still register; `EXACT_EMD` raises a clear
`RuntimeError` at call time.

### Build system

`setup.py` defaults to **JIT-only mode** (no extensions compiled). Opt in
with `BUILD_*` environment variables:

- `TORCHMATCH_BUILD_CPU=1` — builds `torchmatch._assignment_cpu_impl`
  (`-O3 -std=c++17 -mavx2 -mfma` on x86-64).
- `TORCHMATCH_BUILD_CUDA=1` — builds `torchmatch._assignment_cuda_impl`
  (requires `CUDA_HOME`).
- `TORCHMATCH_BUILD_TRANSPORT=1` — builds `torchmatch._transport_cpu_impl`
  (`-O3 -std=c++17 -fno-fast-math`). When combined with `BUILD_CUDA`,
  also builds `torchmatch._transport_cuda_impl`.
- `TORCHMATCH_BUILD_STABLE_ABI=1` — adds `-DPy_LIMITED_API=0x030d0000
-DTORCH_TARGET_VERSION=0x020a000000000000` to all CPU extension builds and
  sets `py_limited_api=True`, producing a `cp313-abi3` wheel that runs with
  any torch ≥ 2.10 and Python ≥ 3.13. CPU-only; not valid with `BUILD_CUDA`.

Default `uv build` (no flags) produces a `py3-none-any` wheel containing
the C++ sources. At first import the JIT path compiles them using the
installed torch, detecting AVX2/FMA on the host CPU at runtime.

All `.so` files land in `site-packages/torchmatch/`.

### Release pipeline (`.github/workflows/release.yml`)

Per-variant wheels (`+cpu`, `+cu126`, `+cu128`, `+cu130`, `+cu132`) build in
manylinux / NVIDIA CUDA containers. `auditwheel repair` exclude list
differs between cu12 and cu13x variants. `scripts/build_pypi_index.py`
generates PEP 503 indexes under `docs/simple/<variant>/torchmatch/`.

### Docs (`docs/site/`)

Nuxt 4 + Nuxt UI + Nuxt Content. Pages under
`docs/site/content/{1.getting-started.md, 2.assignment/**, 3.transport/**,
4.benchmarks/**}`. CI builds via `bun run build` in
`.github/workflows/docs.yml`.

## Code style

- Python: `ruff`, 88-col, target `py313`. Tests exempt from docstring rules.
- C++: `.clang-format` LLVM-based, 88-col, 4-space indent.

## Testing notes

- `tests/conftest.py` exposes `ref_cost` fixture (calls
  `scipy.optimize.linear_sum_assignment`) — the correctness oracle.
- `test_lap_opcheck.py`: `torch.library.opcheck` for schema + FakeTensor
  only. `test_aot_dispatch_dynamic` intentionally omitted (tied costs admit
  multiple equally-correct assignments).
- Timeout: 10 s per test; benchmark files override with
  `pytestmark = pytest.mark.timeout(0)`.
- `tests/_costgen.py`: `uniform`, `gamma`, `iou`, `gated_sparse`,
  `integer_tied` cost distributions for benchmark suites.
