---
title: Transport tutorials
description: Hands-on Jupyter notebooks covering optimal transport — from earth-mover intuition through Sinkhorn and point-cloud Wasserstein losses.
navigation:
  title: Tutorials
---

Three progressive tutorials on the **transport** family, targeting readers
with Python/PyTorch experience but no prior knowledge of optimal transport.

| Tutorial | Topic | Notebook |
|---|---|---|
| [Optimal transport](/algorithms/transport/tutorials/optimal-transport) | Earth-mover intuition, transport plans, Wasserstein distance | `transport/01_optimal_transport.py` |
| [Sinkhorn algorithm](/algorithms/transport/tutorials/sinkhorn) | Entropic regularisation, iterations, divergence | `transport/02_sinkhorn_algorithm.py` |
| [Point clouds and shapes](/algorithms/transport/tutorials/point-clouds) | `samples.loss`, shape generation, unbalanced OT | `transport/03_point_clouds.py` |

## Running the notebooks locally

```bash
# Install the notebooks dependency group
uv sync --extra cu128 --group notebooks

# Launch Jupyter Lab
jupyter lab notebooks/

# Re-render the docs pages from executed notebooks
nix run .#nb-render -- --family transport
```

`transport/03_point_clouds.py` uses `torchmatch.transport.samples.loss`, which
requires a CUDA GPU.  A CPU fallback via `transport.matrix.solve` is provided
for environments without a GPU.

Notebooks are stored as jupytext percent-format `.py` files under
`notebooks/transport/`.
