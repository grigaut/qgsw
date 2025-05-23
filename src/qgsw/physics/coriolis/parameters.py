"""Coriolis Parameters."""

import torch

from qgsw.exceptions import UnitError
from qgsw.physics.constants import EARTH_ANGULAR_ROTATION
from qgsw.spatial.core.grid import Grid2D
from qgsw.utils.units._units import Unit


def compute_coriolis_parameter(
    grid_2d: Grid2D,
) -> torch.Tensor:
    """Compute the coriolis parameter given a 2D Grid.

    Args:
        grid_2d (Grid2D): 2D Grid.

    Raises:
        UnitError: If the grid if not in radians.

    Returns:
        torch.Tensor: (nx, ny) Coriolis parameter value tensor.
    """
    if grid_2d.xy_unit != Unit.RAD:
        msg = f"Unable to compute beta plane from {grid_2d.xy_unit} grid."
        raise UnitError(msg)

    latitude = grid_2d.xy.y  # in radians
    return 2 * EARTH_ANGULAR_ROTATION * torch.sin(latitude)
