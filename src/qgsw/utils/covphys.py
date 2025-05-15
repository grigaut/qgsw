"""Covariant <-> physical conversions."""

from __future__ import annotations

from typing import overload

from qgsw.fields.variables.prognostic_tuples import (
    UVH,
    UVHT,
    BaseUVH,
    UVHTAlpha,
)


@overload
def to_phys(vars_tuple: UVH, dx: float, dy: float) -> UVH: ...
@overload
def to_phys(vars_tuple: UVHT, dx: float, dy: float) -> UVHT: ...
@overload
def to_phys(vars_tuple: UVHTAlpha, dx: float, dy: float) -> UVHTAlpha: ...


def to_phys(
    vars_tuple: BaseUVH,
    dx: float,
    dy: float,
) -> BaseUVH:
    """Convert covariant prognostic tuple to physical.

    Args:
        vars_tuple (BaseUVH): Covariant prognostic tuple.
        dx (float): Elementary distance in the X direction.
        dy (float): Elementary distance in the Y direction.

    Returns:
        BaseUVH: Physical prognostic tuple.
    """
    u, v, h = vars_tuple.uvh
    return vars_tuple.with_uvh(UVH(u / dx, v / dy, h / (dx * dy)))


@overload
def to_cov(vars_tuple: UVH, dx: float, dy: float) -> UVH: ...
@overload
def to_cov(vars_tuple: UVHT, dx: float, dy: float) -> UVHT: ...
@overload
def to_cov(vars_tuple: UVHTAlpha, dx: float, dy: float) -> UVHTAlpha: ...


def to_cov(
    vars_tuple: BaseUVH,
    dx: float,
    dy: float,
) -> BaseUVH:
    """Convert physical prognostic tuple to physical.

    Args:
        vars_tuple (BaseUVH): Physical prognostic tuple.
        dx (float): Elementary distance in the X direction.
        dy (float): Elementary distance in the Y direction.

    Returns:
        BaseUVH: Covariant prognostic tuple.
    """
    u, v, h = vars_tuple.uvh
    return vars_tuple.with_uvh(UVH(u * dx, v * dy, h * (dx * dy)))
