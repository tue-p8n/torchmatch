---
title: Assignment tutorials
description: Hands-on Jupyter notebooks covering the linear assignment problem — from first principles through SORT-style object tracking.
navigation:
  title: Tutorials
---

Three progressive tutorials on the **assignment** family, targeting readers
with Python/PyTorch experience but no prior knowledge of combinatorial
optimisation.

| Tutorial | Topic | Notebook |
|---|---|---|
| [Fundamentals](/algorithms/assignment/tutorials/fundamentals) | What LAP is, why brute force fails, `assignment.solve` | `assignment/01_the_assignment_problem.py` |
| [Backends and batching](/algorithms/assignment/tutorials/backends) | Backend selection, 3-D batched solve, unpacked output | `assignment/02_backends_and_batching.py` |
| [Object tracking](/algorithms/assignment/tutorials/tracking) | IoU cost matrix, SORT-style tracker, trajectory visualisation | `assignment/03_object_tracking.py` |

## Running the notebooks locally

```bash
# Install the notebooks dependency group
uv sync --extra cu128 --group notebooks

# Launch Jupyter Lab
jupyter lab notebooks/

# Re-render the docs pages from executed notebooks
nix run .#nb-render -- --family assignment
```

Notebooks are stored as jupytext percent-format `.py` files under
`notebooks/assignment/`.  `nix run .#nb-render` converts them to `.ipynb`,
executes each one, and writes the output to these pages.
