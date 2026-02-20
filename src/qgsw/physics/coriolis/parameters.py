"""Coriolis Parameters."""

import torch

from qgsw.physics.constants import EARTH_ANGULAR_ROTATION
from qgsw.spatial.core.grid import Grid2D


def compute_coriolis_parameter(
    grid_2d: Grid2D,
) -> torch.Tensor:
    """Compute the coriolis parameter given a 2D Grid.

    Args:
        grid_2d (Grid2D): 2D Grid.


    Returns:
        torch.Tensor: (nx, ny) Coriolis parameter value tensor.
    """
    latitude = grid_2d.xy.y  # in radians
    return 2 * EARTH_ANGULAR_ROTATION * torch.sin(latitude)
