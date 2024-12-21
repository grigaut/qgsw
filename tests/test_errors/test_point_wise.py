"""Test point-wise errors."""

import pytest
import torch

from qgsw.fields.errors.point_wise import RMSE
from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    Pressure,
    SurfaceHeightAnomaly,
)
from qgsw.fields.variables.uvh import UVH


@pytest.fixture
def pressure() -> Pressure:
    """Pressure."""
    g_prime = (
        torch.tensor([10, 0.05], dtype=torch.float64, device="cpu")
        .unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    h_phys = PhysicalLayerDepthAnomaly(ds=1)
    eta = SurfaceHeightAnomaly(h_phys)
    return Pressure(g_prime, eta)


def test_RMSE(pressure: Pressure) -> None:  # noqa: N802
    """Test RMSE."""
    n_ens = 1
    nl = 2
    nx = 8
    ny = 12

    rmse = RMSE(pressure, pressure)
    uvh = UVH(
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
    )

    assert rmse.compute_point_wise(uvh, uvh).shape == (n_ens, nl, nx, ny)
    assert rmse.compute_level_wise(uvh, uvh).shape == (n_ens, nl)
    assert rmse.compute_ensemble_wise(uvh, uvh).shape == (n_ens,)

    assert (rmse.compute_point_wise(uvh, uvh) == 0).all()
    assert (rmse.compute_level_wise(uvh, uvh) == 0).all()
    assert (rmse.compute_ensemble_wise(uvh, uvh) == 0).all()

    uvh_ref = UVH(
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
        torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu"),
    )

    p = pressure.compute(uvh)
    p_ref = pressure.compute(uvh_ref)
    rmse_ref = torch.sqrt((p - p_ref) ** 2)

    assert (rmse.compute_point_wise(uvh, uvh_ref) == rmse_ref).all()
