"""Schemes for Time-integration."""

from collections.abc import Callable

from qgsw.fields.variables.uvh import UVH


def rk3_ssp(
    uvh: UVH,
    dt: float,
    time_derivation_func: Callable[[UVH], UVH],
) -> UVH:
    """Perform time-integration using a RK3-SSP scheme..

    Args:
        uvh (UVH): Prognostic variables.
        dt (float): Timestep.
        time_derivation_func (Callable): Time derivation.

    Returns:
        UVH: Final uvh variables.
    """
    dt0_uvh = time_derivation_func(uvh)
    uvh += dt * dt0_uvh

    dt1_uvh = time_derivation_func(uvh)
    uvh += (dt / 4) * (dt1_uvh - 3 * dt0_uvh)

    dt2_uvh = time_derivation_func(uvh)
    uvh += (dt / 12) * (8 * dt2_uvh - dt1_uvh - dt0_uvh)

    return uvh
