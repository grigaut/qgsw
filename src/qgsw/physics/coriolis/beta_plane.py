"""Beta-Plane compuations."""

import torch

from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.spatial.conversion import deg_to_m_lat, deg_to_rad, km_to_m
from qgsw.spatial.core.mesh import Mesh2D
from qgsw.spatial.units._units import DEGREES, KILOMETERS, METERS, RADIANS
from qgsw.spatial.units.exceptions import UnitError


def compute_entire_beta_plane(
    mesh_2d: Mesh2D,
) -> torch.Tensor:
    """Compute Beta-Plane field given only a 2D Mesh.

    Args:
        mesh_2d (Mesh2D): 2D Mesh.

    Raises:
        UnitError: If the mesh xy unit is invalid.

    Returns:
        torch.Tensor: Beta plane values in s⁻¹.
    """
    if mesh_2d.xy_unit == DEGREES:
        return _beta_plane_from_degree(
            latitude=mesh_2d.xy[1],
            f0=compute_f0(mesh_2d=mesh_2d),
            beta=compute_beta(mesh_2d=mesh_2d),
        )
    if mesh_2d.xy_unit == RADIANS:
        return _beta_plane_from_radians(
            latitude=mesh_2d.xy[1],
            f0=compute_f0(mesh_2d=mesh_2d),
            beta=compute_beta(mesh_2d=mesh_2d),
        )
    msg = f"Unable to compute beta plane from unit {mesh_2d.xy_unit}."
    raise UnitError(msg)


def compute_beta_plane(
    mesh_2d: Mesh2D,
    f0: float,
    beta: float,
) -> torch.Tensor:
    """Compute beta-plane from a given mesh_2d.

    Args:
        mesh_2d (Mesh2D): 2D Mesh to compute values for.
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    if mesh_2d.xy_unit == METERS:
        return _beta_plane_from_meters(y=mesh_2d.xy.y, f0=f0, beta=beta)
    if mesh_2d.xy_unit == KILOMETERS:
        y = km_to_m(mesh_2d.xy.y)
        return _beta_plane_from_meters(y=y, f0=f0, beta=beta)
    if mesh_2d.xy_unit == DEGREES:
        latitude = mesh_2d.xy.y
        return _beta_plane_from_degree(latitude=latitude, f0=f0, beta=beta)
    if mesh_2d.xy_unit == RADIANS:
        latitude = mesh_2d.xy.y
        return _beta_plane_from_radians(latitude=latitude, f0=f0, beta=beta)
    msg = f"Unable to compute beta plane from unit {mesh_2d.xy_unit}."
    raise UnitError(msg)


def _beta_plane_from_meters(
    y: torch.Tensor,
    f0: float,
    beta: float,
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
    latitude: torch.Tensor,
    f0: float,
    beta: float,
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
    latitude: torch.Tensor,
    f0: float,
    beta: float,
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


def compute_f0(mesh_2d: Mesh2D) -> float:
    """Compute f0 value given 2D Mesh.

    Args:
        mesh_2d (Mesh2D): Mesh to compute f0 from.

    Raises:
        UnitError: If the mesh unit is invalid.

    Returns:
        float: f0 value (value at the mean latitude) in s⁻¹.
    """
    if mesh_2d.xy_unit == RADIANS:
        return _compute_f0_from_radians(latitude_ref=mesh_2d.xy.y.mean())
    if mesh_2d.xy_unit == DEGREES:
        latitude_ref = deg_to_rad(mesh_2d.xy.y.mean())
        return _compute_f0_from_radians(latitude_ref=latitude_ref)
    msg = f"Unable to compute f0 using a {mesh_2d.xy_unit} mesh."
    raise UnitError(msg)


def _compute_f0_from_radians(
    latitude_ref: float,
) -> float:
    """Compute f0 parameter given a reference latitude.

    Args:
        latitude_ref (float): Reference latitude (radians).

    Returns:
        float: f0 value in s⁻¹.
    """
    return 2 * EARTH_ANGULAR_ROTATION * torch.sin(latitude_ref)


def compute_beta(mesh_2d: Mesh2D) -> float:
    """Compute beta value given 2D Mesh.

    Args:
        mesh_2d (Mesh2D): Mesh to compute beta from.

    Raises:
        UnitError: If the mesh unit is invalid.

    Returns:
        float: beta value (value at the mean latitude) in m⁻¹.s⁻¹.
    """
    if mesh_2d.xy_unit == RADIANS:
        return _compute_beta_from_radians(latitude_ref=mesh_2d.xy.y.mean())
    if mesh_2d.xy_unit == DEGREES:
        latitude_ref = deg_to_rad(mesh_2d.xy.y.mean())
        return _compute_beta_from_radians(latitude_ref=latitude_ref)
    msg = f"Unable to compute beta using a {mesh_2d.xy_unit} mesh."
    raise UnitError(msg)


def _compute_beta_from_radians(
    latitude_ref: float,
) -> float:
    """Compute f0 parameter given a reference latitude.

    Args:
        latitude_ref (float): Reference latitude (radians).

    Returns:
        float: beta value in m⁻¹.s⁻¹.
    """
    return (
        2
        * EARTH_ANGULAR_ROTATION
        * torch.cos(torch.mean(latitude_ref))
        / EARTH_RADIUS
    )
