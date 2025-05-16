"""Test point-wise errors."""

import pytest
import torch

from qgsw.fields.errors.point_wise import RMSE
from qgsw.fields.variables.physical import (
    Pressure,
    SurfaceHeightAnomaly,
)
from qgsw.fields.variables.tuples import UVH
from qgsw.specs import DEVICE


@pytest.fixture
def pressure() -> Pressure:
    """Pressure."""
    g_prime = (
        torch.tensor([10, 0.05], dtype=torch.float64, device=DEVICE.get())
        .unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    eta_phys = SurfaceHeightAnomaly()
    return Pressure(g_prime, eta_phys)


def test_RMSE(pressure: Pressure) -> None:  # noqa: N802
    """Test RMSE."""
    n_ens = 1
    nl = 2
    nx = 8
    ny = 12

    rmse = RMSE(pressure, pressure)
    uvh = UVH(
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
    )

    assert (rmse.compute_point_wise(uvh, uvh) == 0).all()
    assert (rmse.compute_level_wise(uvh, uvh) == 0).all()
    assert (rmse.compute_ensemble_wise(uvh, uvh) == 0).all()

    uvh_ref = UVH(
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
    )

    p = pressure.compute(uvh)
    p_ref = pressure.compute(uvh_ref)
    rmse_ref = torch.sqrt((p - p_ref) ** 2)

    assert (rmse.compute_point_wise(uvh, uvh_ref) == rmse_ref).all()


def test_error_slicing(pressure: Pressure) -> None:
    """Test error slicing."""
    n_ens = 1
    nl = 2
    nx = 8
    ny = 12

    rmse = RMSE(pressure, pressure)
    uvh = UVH(
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
    )
    uvh_ref = UVH(
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.rand(
            (n_ens, nl, nx, ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
    )

    point_wise = rmse.compute_point_wise(uvh, uvh_ref)
    level_wise = rmse.compute_level_wise(uvh, uvh_ref)
    ensemble_wise = rmse.compute_ensemble_wise(uvh, uvh_ref)

    assert point_wise.shape == (n_ens, nl, nx, ny)
    assert level_wise.shape == (n_ens, nl)
    assert ensemble_wise.shape == (n_ens,)

    slices = [slice(0, 1), slice(0, 1), slice(0, 5), ...]

    rmse.slices = slices

    assert rmse.compute_point_wise(uvh, uvh_ref).shape == (n_ens, 1, 5, ny)
    assert rmse.compute_level_wise(uvh, uvh_ref).shape == (n_ens, 1)
    assert rmse.compute_ensemble_wise(uvh, uvh_ref).shape == (n_ens,)

    assert (
        rmse.compute_point_wise(uvh, uvh_ref) == point_wise.__getitem__(slices)
    ).all()

    rmse.slices = [...]

    assert rmse.compute_point_wise(uvh, uvh_ref).shape == (n_ens, nl, nx, ny)
    assert rmse.compute_level_wise(uvh, uvh_ref).shape == (n_ens, nl)
    assert rmse.compute_ensemble_wise(uvh, uvh_ref).shape == (n_ens,)
