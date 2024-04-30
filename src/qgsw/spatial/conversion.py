"""Unit Conversion Tools."""

import numpy as np

DEG_TO_M = 111e3  # π * 6371e3 / 180
KM_TO_M = 1e3


def deg_to_m_lat(value: float) -> float:
    """Degree to Meters conversion.

    Args:
        value (float): Value in degree.

    Returns:
        float: Value in Meters.
    """
    return value * DEG_TO_M


def m_to_deg_lat(value: float) -> float:
    """Meters to degree conversion for latitudes.

    Args:
        value (float): Value in meters.

    Returns:
        float: Value in degree.
    """
    return value / DEG_TO_M


def km_to_m(value: float) -> float:
    """Kilometers to Meters conversion.

    Args:
        value (float): Value in kilometers.

    Returns:
        float: Value in meters.
    """
    return value * KM_TO_M


def m_to_m(value: float) -> float:
    """Meters to Meters conversion.

    Args:
        value (float): Value in meters.

    Returns:
        float: Value in meters.
    """
    return value


def rad_to_deg(value: float) -> float:
    """Radians to degrees conversion.

    Args:
        value (float): Value in radians.

    Returns:
        float: Value in degrees.
    """
    return value * 180 / np.pi


def deg_to_rad(value: float) -> float:
    """Degrees to radians conversion.

    Args:
        value (float): Value in degree.

    Returns:
        float: Value in radians.
    """
    return value * np.pi / 180


def rad_to_m(value: float) -> float:
    """Radians to meters conversion.

    Args:
        value (float): Value in radians.

    Returns:
        float: Value in meters.
    """
    return deg_to_m_lat(rad_to_deg(value))
