"""Fields computations."""

import xarray as xr

from qgsw.eNATL60.var_keys import (
    TIME,
)


def compute_streamfunction_with_atmospheric_pressure(
    da_ssh: xr.DataArray,
    da_atmp: xr.DataArray,
    rho0: float = 1026.0,
    g: float = 9.81,
    *,
    remove_avgs: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        da_ssh (xr.DataArray): Sea surface height.
        da_atmp (xr.DataArray): Atmospheric pressure.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avgs (bool, optional): Whether to remove atmospheric pressure
            average (spatial average evaluated at every time) and ssh average
            (both in time and space) or not. Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    if remove_avgs:
        atmos_p_avg = da_atmp.mean(dim=[d for d in da_atmp.dims if d != TIME])
        return (da_atmp - atmos_p_avg) / rho0 + g * (da_ssh - da_ssh.mean())
    return da_atmp / rho0 + g * da_ssh


def compute_streamfunction_with_atmospheric_pressure_txy_avg(
    da_ssh: xr.DataArray,
    da_atmp: xr.DataArray,
    rho0: float = 1026.0,
    g: float = 9.81,
    *,
    remove_avgs: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        da_ssh (xr.DataArray): Sea surface height.
        da_atmp (xr.DataArray): Atmospheric pressure.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avgs (bool, optional): Whether to remove atmospheric pressure
            average (spatial average evaluated at every time) and ssh average
            (both in time and space) or not. Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    if remove_avgs:
        return (da_atmp - da_atmp.mean()) / rho0 + g * (da_ssh - da_ssh.mean())

    return (da_atmp) / rho0 + g * da_ssh


def compute_streamfunction_with_atmospheric_pressure_xy_avg(
    da_ssh: xr.DataArray,
    da_atmp: xr.DataArray,
    rho0: float = 1026.0,
    g: float = 9.81,
    *,
    remove_avgs: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        da_ssh (xr.DataArray): Sea surface height.
        da_atmp (xr.DataArray): Atmospheric pressure.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avgs (bool, optional): Whether to remove atmospheric pressure
            average (spatial average evaluated at every time) and ssh average
            (both in time and space) or not. Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    if remove_avgs:
        atmos_p_avg = da_atmp.mean(dim=[d for d in da_atmp.dims if d != TIME])
        ssh_avg = da_ssh.mean(dim=[d for d in da_ssh.dims if d != TIME])
        return (da_atmp - atmos_p_avg) / rho0 + g * (da_ssh - ssh_avg)
    return da_atmp / rho0 + g * da_ssh


def compute_stream_function_ssh_only(
    da_ssh: xr.DataArray,
    g: float = 9.81,
    *,
    remove_avg: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        da_ssh (xr.DataArray): Sea surface height.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avg (bool, optional): Whether to remove ssh average or not.
            Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    if remove_avg:
        return g * (da_ssh - da_ssh.mean())
    return g * da_ssh
