"""Parameterizations."""

import torch


def kappa_to_alpha(kappa: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Kappa to Alpha parameterization.

    Args:
        kappa (torch.Tensor): Kappa
        epsilon (float, optional): Epsilon. Defaults to 0.1.

    Returns:
        torch.Tensor: Alpha = e^(κ*(|κ|+ϵ))-1
    """
    return torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1
