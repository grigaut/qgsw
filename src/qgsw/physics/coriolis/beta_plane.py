"""Beta-Plane compuations."""

import torch

from qgsw.mesh.mesh import Mesh2D
from qgsw.physics.constants import EARTH_RADIUS
from qgsw.spatial.conversion import deg_to_m_lat, km_to_m
from qgsw.spatial.units._units import DEGREES, KILOMETERS, METERS, RADIANS
from qgsw.spatial.units.exceptions import UnitError


def compute_beta_plane(
    mesh: Mesh2D,
    f0: float,
    beta: float,
) -> torch.Tensor:
    """Compute beta-plane from a given mesh.

    Args:
        mesh (Mesh2D): 2D Mesh to compute values for.
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    if mesh.xy_unit == METERS:
        return _beta_plane_from_meters(y=mesh.xy[1], f0=f0, beta=beta)
    if mesh.xy_unit == KILOMETERS:
        return _beta_plane_from_meters(y=km_to_m(mesh.xy[1]), f0=f0, beta=beta)
    if mesh.xy_unit == DEGREES:
        return _beta_plane_from_degree(latitude=mesh.xy[1], f0=f0, beta=beta)
    if mesh.xy_unit == RADIANS:
        return _beta_plane_from_radians(latitude=mesh.xy[1], f0=f0, beta=beta)
    msg = f"Unable to compute beta plane from unit {mesh.xy_unit}."
    raise UnitError(msg)


def _beta_plane_from_meters(
    y: torch.Tensor, f0: float, beta: float
) -> torch.Tensor:
    """Compute beta-plane from y in meters.

    Args:
        y (torch.Tensor): y values (meters).
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    return f0 + beta * (y - y.mean())


def _beta_plane_from_degree(
    latitude: torch.Tensor, f0: float, beta: float
) -> torch.Tensor:
    """Compute beat-plane from latitudes in degrees.

    Args:
        latitude (torch.Tensor): latitude values (degrees).
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    return f0 + beta * deg_to_m_lat(latitude - latitude.mean())


def _beta_plane_from_radians(
    latitude: torch.Tensor, f0: float, beta: float
) -> torch.Tensor:
    """Compute beat-plane from latitudes in radians.

    Args:
        latitude (torch.Tensor): latitude values (radians).
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    return f0 + beta * (latitude - latitude.mean()) * EARTH_RADIUS
