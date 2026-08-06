<img src="https://torchmatch.khws.io/logo.svg" alt="torchmatch" height="40" />

Solvers for assignment and optimal-transport problems on top of
PyTorch. Call `torchmatch.assignment.solve(cost)` or
`torchmatch.transport.matrix.solve(cost)` and ship, or call a
specific op when you want to choose the kernel. The ops share the
`torch.ops.assignment` and `torch.ops.transport` namespaces, both
trace under `torch.compile`, and both run on CPU and CUDA (the
`transport.samples` face is CUDA-only).

**Documentation**: <https://torchmatch.khws.io/>. Tutorials,
API reference, and the full benchmark report.

## Install

### Standard Installation (PyPI)
```bash
pip install torchmatch
```
By default, this installs the CPU stable-ABI wheel or the source distribution (`sdist`). If a CUDA device is present, `import torchmatch` will automatically JIT-compile the CUDA extensions at import-time (cached for subsequent runs).

### Pre-compiled CUDA Wheels
To bypass JIT compilation on GPU machines, install a precompiled CUDA-variant wheel (`+cu126`, `+cu128`, `+cu130`, `+cu132`) from our PEP 503 package index:

**With `pip`**:
```bash
pip install torchmatch --extra-index-url https://torchmatch.khws.io/simple/cu128/
```

**With `uv`** (configure in `pyproject.toml`):
```toml
[[tool.uv.index]]
name = "torchmatch-cu128"
url = "https://torchmatch.khws.io/simple/cu128/"
explicit = true

[tool.uv.sources]
torchmatch = { index = "torchmatch-cu128" }
```
Or add via CLI:
```bash
uv add torchmatch --index torchmatch-cu128=https://torchmatch.khws.io/simple/cu128/
```

## Usage

`torchmatch.assignment.solve` validates the input, picks a backend by
device and shape, and returns a tensor whose rank is `cost.ndim - 1`:

```python
import torch
from torchmatch.assignment import solve

cost = torch.rand(32, 32)
row_to_col = solve(cost)                  # CPU, picks a JV variant
row_to_col_cuda = solve(cost.cuda())      # CUDA, picks a primed-zeros op

batched = solve(torch.rand(8, 16, 16))    # (B, N) packed result
```

The individual ops remain exported for callers who want kernel-level
control or are benchmarking:

```python
from torchmatch.assignment.ops import jonker_dense, lawler, munkres

jonker_dense(cost)                       # explicit op, CPU
munkres(cost.cuda())                     # explicit op, CUDA
lawler(cost.cuda())                      # explicit op, CUDA
```

Pass `backend=` to override the dispatcher
(`torchmatch.assignment.Backend.AUTO` / `.JONKER` / `.MUNKRES` /
`.LAWLER` / `.GREEDY`), and `unpack=True` to get matched pairs and
unmatched rows / cols as a tuple instead of a single mapping. Every
op returns an `int64` row->col mapping of length `N`; `-1` marks an
unmatched row.

The same ops live at `torch.ops.assignment.<op>`, useful inside
`torch.compile` regions or when dispatching by name. To control
extension loading (preforked workers, latency-sensitive serving), the
per-family loaders stay exported and idempotent:

```python
from torchmatch.assignment import load_cpu, load_cuda

load_cpu()                               # already ran at import; no-op
load_cuda()
```

## Transport

The sibling `torchmatch.transport` namespace ships continuous
optimal-transport solvers under two sub-packages.

The matrix face takes a `(N, M)` or `(B, N, M)` cost matrix and
returns a transport plan (or scalar divergence). Four backends:
`LOG_SINKHORN` (default), `SINKHORN_DIVERGENCE`,
`UNBALANCED_SINKHORN`, and `EXACT_EMD` (network simplex, CPU-only).
All other backends run on both CPU and CUDA.

```python
import torch
from torchmatch.transport.matrix import Backend, solve

cost = torch.rand(64, 64)
log_plan = solve(cost, reg=0.1, n_iter=200)            # default LOG_SINKHORN
plan_exact = solve(cost, backend=Backend.EXACT_EMD)    # CPU-only
```

The samples face takes two point clouds `(N, D)` and `(M, D)` and
returns a scalar OT loss; squared-Euclidean cost is computed on the
fly via Triton kernels. CUDA-only.

```python
from torchmatch.transport.samples import loss

x = torch.randn(1024, 8, device="cuda")
y = torch.randn(1024, 8, device="cuda")
sloss = loss(x, y, blur=0.05)
sdiv  = loss(x, y, blur=0.05, debias=True)             # Sinkhorn divergence
sunb  = loss(x, y, blur=0.05, reach=1.0)               # unbalanced
```

The individual matrix-face ops are exported for kernel-level control:

```python
from torchmatch.transport.matrix.ops import (
    log_sinkhorn, sinkhorn_divergence, unbalanced_sinkhorn, exact_emd,
)

log_sinkhorn(cost.unsqueeze(0), 0.1, 200, a, b, None, None)
```

The same ops live at `torch.ops.transport.<op>`. The per-family loaders
stay exported and idempotent for preforked workers and latency-sensitive
serving:

```python
from torchmatch.transport import load_cpu, load_cuda

load_cpu()
load_cuda()
```

See the [documentation](https://torchmatch.khws.io/) for full API reference.

## Ops

### CUDA primed-zeros Hungarian ops

Two implementations within the primed/starred-zeros sub-family of
the Hungarian method. `munkres` is Munkres' (1957) single-path
augmenting-path variant; `lawler` is Lawler's (1976) tree-augmentation
variant. Both are CUDA-only, and both raise a clear error if called
inside CUDA graph capture, since host-side syncs read managed-memory
flags — see [torch.compile / torch.export](#torchcompile--torchexport)
below. The JV CUDA backend of `jonker_dense_batch` is also a Hungarian
op, documented in the JV section below.

| Op        | dtype (internal)      | Variant                                                                                                                                                   |
| --------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `munkres` | float32, column-major | Munkres' classical single-path augmenting-path Hungarian; CUB segmented reductions for column-min. Sparse-favored.                                        |
| `lawler`  | float64, row-major    | Lawler's tree augmentation: parallel BFS finds all vertex-disjoint augmenting paths per outer iteration; cooperative-groups + Thrust scan. Dense-favored. |

### Jonker-Volgenant family, CPU (single problem)

Successive-shortest-path (Dijkstra-like over reduced costs). All accept
rectangular `(N, M)` cost matrices in float32 or float64.

| Op               | Variant                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `jonker_scalar`  | Sequential reference; no SIMD. Implements Crouse 2016. Rejects NaN / -inf; treats +inf as an infeasible edge. |
| `jonker_dense`   | AVX2 flat-pointer inner loop; rectangular-capable                                                             |
| `jonker_compact` | AVX2-gather inner loop; square-only internal kernel (rectangular inputs padded by the wrapper)                |

### Jonker-Volgenant family, batched

| Op                              | Devices    | Constraints                                                                                                                                                                                    |
| ------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `jonker_dense_batch`            | CPU + CUDA | CPU: `(B, N, M)`, any size, `at::parallel_for` over per-problem `jonker_dense`. CUDA: shared-memory tiled kernel, one block per problem; requires `(B, K, K)` square with `K ≤ MAX_TILE = 64`. |
| `jonker_compact_batch`          | CPU        | `at::parallel_for` over per-problem `jonker_compact`                                                                                                                                           |
| `jonker_dense_batch_unpacked`   | CPU        | Returns `(matches, unmatched_rows, unmatched_cols, n_matched)`, which saves a per-problem Python unpack                                                                                        |
| `jonker_compact_batch_unpacked` | CPU        | Same shape, compact variant                                                                                                                                                                    |

## torch.compile / torch.export

Every op carries a FakeTensor kernel. All four CUDA-only assignment ops
(`munkres`, `hybrid`, `lawler`, and the CUDA backend of
`jonker_dense_batch`) do a host-side CUDA synchronization internally and
raise a clear `RuntimeError` if called inside CUDA graph capture (e.g.
under `torch.compile(mode="reduce-overhead")`) instead of graph-breaking
around it — call them outside the compiled/captured region.

## Build modes

The per-device loaders prefer a prebuilt `.so` shipped in the wheel
and fall back to JIT-compiling the C++/CUDA sources via
`torch.utils.cpp_extension.load`. Both paths register the same
`torch.ops.assignment.*` ops; the choice is invisible to callers.

### Building wheels

```bash
# default: builds CPU extension; builds CUDA extension if a CUDA
# toolchain is detected (torch.utils.cpp_extension.CUDA_HOME)
pip wheel . -w dist/

# CPU-only wheel
TORCHMATCH_SKIP_CUDA=1 pip wheel . -w dist/

# CUDA SM targets (default: PyTorch's current-device default)
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" pip wheel . -w dist/
```

The build system pairs `setuptools` with `torch.utils.cpp_extension.BuildExtension`.
Sdists ship the full `sources/torchmatch/{assignment,transport}/{cpu,cuda}/`
tree so the JIT path works on any CUDA-capable machine without a
matching wheel.

### Runtime overrides

- `TORCHMATCH_FORCE_JIT=1`: skip the prebuilt `.so` and recompile from
  source via `cpp_extension.load`. Useful during development and for
  diagnosing ABI mismatches.

## Development environment

The `flake.nix` pins Python 3.13, uv, and a chosen CUDA toolkit. Dev
shells build the variant venv from `uv.lock` via `uv2nix`, so the Python
environment is ready without running `uv sync`. With direnv:

```bash
direnv allow                      # picks devShells.default = cu128
NIX_DEVSHELL_NAME=cpu  direnv reload
NIX_DEVSHELL_NAME=cu130 direnv reload
```

Without direnv:

```bash
nix develop                       # default = cu128
nix develop .#cpu                 # CPU-only (sets TORCHMATCH_SKIP_CUDA=1)
nix develop .#cu126
nix develop .#cu130
```

Once inside the shell, the variant venv is already on PATH:

```bash
python -m pytest tests/           # run the test suite
uv sync --all-groups              # only needed for editable iteration on torchmatch
```

Every project task is a flake app. The surface lives in `nix/apps.nix`
and replaces the previous `justfile`:

```bash
nix flake show                    # enumerate every app / devShell / package
nix run .#test                    # default variant (cu128)
nix run .#test-cpu                # pin a different torch ABI
nix run .#lint                    # ruff check .
nix run .#format                  # ruff format .
                                  # On CPU-only hosts, prefer `.#test-cpu` /
                                  # `.#lint-cpu` / `.#format-cpu` to avoid
                                  # pulling the CUDA wheel closure.
nix run .#benchmark-init              # one-time machine registration
nix run .#benchmark-collect           # run the benchmark sweep
nix run .#benchmark-aggregate         # build the static dashboard datasets
nix run .#benchmark-validate          # PR-equivalent schema check
nix run .#docs-serve              # Nuxt dev server at 127.0.0.1:3000
nix run .#docs-build              # static build of docs/site/
nix run .#docs-preview            # build + serve via python -m http.server
```

For a Nix-built artifact of the C++/CUDA extension:

```bash
nix build .#torchmatch-cpu        # CPU only
nix build .#torchmatch-cu128      # cu128 + torch 2.11
nix build .#torchmatch-cu130      # cu130 + torch 2.12+
```

The result tree contains `result/lib/python3.13/site-packages/torchmatch/`
with the per-family extension `.so` files (`_assignment_cpu_impl*.so`,
`_transport_cpu_impl*.so`, and the matching `_*_cuda_impl*.so` on cuXXX
variants). These artifacts are intended for vendoring or for serving
from a Nix binary cache. For an interactive Python session that
imports the CUDA extension, use `nix develop` (the dev shell exposes
the host's libcuda via `/run/opengl-driver/lib`); the package
derivations do not bundle the CUDA runtime libraries.

The PyPI wheels are still built by the manylinux / NVIDIA CUDA
container pipeline in `.github/workflows/release.yml`; the
`nix build .#torchmatch-*` outputs above are independent of that
pipeline and intended for local reproducibility.
