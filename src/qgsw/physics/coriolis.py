"""Coriolis-related tools."""

import torch


def compute_beta_plane(
    latitudes: torch.Tensor, f0: float, beta: float, ly: float
) -> torch.Tensor:
    """Generate Coriolis Parameter Mesh.

    Args:
        latitudes (torch.Tensor): Latitudes.
        f0 (float): f0 (from beta-plane approximation).
        beta (float): Beta (from beta plane approximation).
        ly (float): Latitudes span.

    Returns:
        torch.Tensor: Coriolis Mesh.
    """
    return f0 + beta * (latitudes - ly / 2)
