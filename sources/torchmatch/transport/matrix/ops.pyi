import torch

def exact_emd(
    cost: torch.Tensor,
    mask: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
) -> torch.Tensor: ...
def log_sinkhorn(
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor: ...
def sinkhorn_divergence(
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor: ...
def unbalanced_sinkhorn(
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    rho: float,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor: ...
