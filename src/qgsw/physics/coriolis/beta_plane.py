"""Beta-Plane compuations."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.spatial.conversion import deg_to_m_lat

if TYPE_CHECKING:
    from qgsw.spatial.core.grid import Grid2D


class BetaPlane(NamedTuple):
    """Beta Plane : f  = f0 + βy."""

    f0: float
    beta: float

    def compute_over_grid(self, grid_2d: Grid2D) -> torch.Tensor:
        """Compute Coriolis Values over a given grid."""
        return compute_beta_plane(
            grid_2d=grid_2d,
            f0=self.f0,
            beta=self.beta,
        )


def compute_beta_plane(
    grid_2d: Grid2D,
    f0: float,
    beta: float,
) -> torch.Tensor:
    """Compute beta-plane from a given grid_2d.

    Args:
        grid_2d (Grid2D): 2D Grid to compute values for.
        f0 (float): f0 (from beta-plane approximation: s⁻¹).
        beta (float): Beta (from beta plane approximation: m⁻¹.s⁻¹).

    Returns:
        torch.Tensor: Coriolis  values.
    """
    return _beta_plane_from_meters(y=grid_2d.xy.y, f0=f0, beta=beta)


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


def compute_f0(grid_2d: Grid2D) -> float:
    """Compute f0 value given 2D Grid.

    Args:
        grid_2d (Grid2D): Grid to compute f0 from.

    Raises:
        UnitError: If the grid unit is invalid.

    Returns:
        float: f0 value (value at the mean latitude) in s⁻¹.
    """
    return _compute_f0_from_radians(latitude_ref=grid_2d.xy.y.mean())


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


def compute_beta(grid_2d: Grid2D) -> float:
    """Compute beta value given 2D Grid.

    Args:
        grid_2d (Grid2D): Grid to compute beta from.

    Raises:
        UnitError: If the grid unit is invalid.

    Returns:
        float: beta value (value at the mean latitude) in m⁻¹.s⁻¹.
    """
    return _compute_beta_from_radians(latitude_ref=grid_2d.xy.y.mean())


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
