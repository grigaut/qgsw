"""Dimensionless numbers computation."""

import numpy as np


def compute_rossby(
    velocity_scale: float,
    f0: float,
    length_scale: float,
) -> float:
    """Compute Rossby Number (Ro).

    Args:
        velocity_scale (float): Velocity scale, unit: m.s⁻¹.
        f0 (float): Coriolis parameter, unit: s⁻¹.
        length_scale (float): Length scale, unit: m.

    Returns:
        float: Rossby number: U/(f * L).
    """
    return velocity_scale / (f0 * length_scale)


def compute_burger(
    g: float, h_scale: float, f0: float, length_scale: float
) -> float:
    """Compute Burger Number (Bu).

    Args:
        g (float): Gravity acceleration, unit: m.s⁻².
        h_scale (float): Vertical length scale, unit: m.
        f0 (float): Coriolis parameter, unit: s⁻¹.
        length_scale (float): Length scale, unit: m.

    Returns:
        float: Burger number: (g * H) / (f0² * Ld²)
    """
    return (g * h_scale) / (f0**2 * length_scale**2)


def compute_deformation_radius(g: float, h_scale: float, f0: float) -> float:
    """Compute Deformation Radius (Rd).

    Args:
        g (float): Gravity acceleration, unit: m.s⁻².
        h_scale (float): Vertical length scale, unit: m.
        f0 (float): Coriolis parameter, unit: s⁻¹.

    Returns:
        float: Deformation Radius: √(g * H) / f0
    """
    return np.sqrt(g * h_scale) / f0
