"""Coriolis Parameters."""

import torch

from qgsw.physics.constants import EARTH_ANGULAR_ROTATION
from qgsw.spatial.core.mesh import Mesh2D
from qgsw.spatial.units._units import RADIANS
from qgsw.spatial.units.exceptions import UnitError


def compute_coriolis_parameter(
    mesh: Mesh2D,
) -> torch.Tensor:
    """Compute the coriolis parameter given a 2D Mesh.

    Args:
        mesh (Mesh2D): 2D Mesh.

    Raises:
        UnitError: If the mesh if not in radians.

    Returns:
        torch.Tensor: (nx, ny) Coriolis parameter value tensor.
    """
    if mesh.xy_unit != RADIANS:
        msg = f"Unable to compute beta plane from {mesh.xy_unit} mesh."
        raise UnitError(msg)

    latitude = mesh.xy[1]  # in radians
    return 2 * EARTH_ANGULAR_ROTATION * torch.sin(latitude)
