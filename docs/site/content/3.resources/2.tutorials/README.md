# torchmatch notebooks

Interactive Jupyter tutorials for the torchmatch library, targeting readers with
post-BSc Python/PyTorch experience but no prior knowledge of the linear assignment
problem or optimal transport.

## Setup

```bash
# Install the notebooks dependency group into the dev environment
uv sync --extra cu128 --group notebooks

# Launch Jupyter Lab
jupyter lab notebooks/
```

## Rendering to docs

The render script executes each notebook and exports a markdown page to
`docs/site/content/<family>/tutorials/`. Figures land in
`docs/site/public/notebooks/<family>/`.

```bash
nix run .#nb-render             # render all (requires CUDA for transport/03)
nix run .#nb-render -- --family assignment
nix run .#nb-render -- --family transport
nix run .#nb-render -- --no-execute   # convert without re-executing
```

Notebooks are standard `.ipynb` files. Open them directly in Jupyter Lab or VS Code.

## Assignment series

| Notebook                                  | Topic                                                             |
| ----------------------------------------- | ----------------------------------------------------------------- |
| `assignment/01_the_assignment_problem.py` | Problem definition, brute-force impossibility, `assignment.solve` |
| `assignment/02_backends_and_batching.py`  | Backend selection, batched 3-D solve, unpacked output             |
| `assignment/03_object_tracking.py`        | IoU cost matrix, SORT-style tracker, trajectory visualisation     |

## Transport series

| Notebook                             | Topic                                                        |
| ------------------------------------ | ------------------------------------------------------------ |
| `transport/01_optimal_transport.py`  | Earth-mover intuition, transport plans, Wasserstein distance |
| `transport/02_sinkhorn_algorithm.py` | Entropic regularisation, Sinkhorn iterations, divergence     |
| `transport/03_point_clouds.py`       | `samples.loss`, shape generation, unbalanced OT              |

## Notes

- Notebooks are stored as **jupytext percent-format `.py` files** (human-readable,
  diff-friendly). `nb-render` converts them to `.ipynb` before executing.
- `transport/03_point_clouds.py` requires a CUDA GPU for the samples face.
  The matrix-face fallback runs on CPU but is slower.
- Figures are saved under `notebooks/figures/` during execution and copied
  to `docs/site/public/notebooks/<family>/` by `nb-render`.
