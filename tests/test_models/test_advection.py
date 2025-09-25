"""Test for advection related functions."""

import pytest
import torch

from qgsw.models.core.flux import div_flux_5pts, div_flux_5pts_only
from qgsw.solver.finite_diff import grad_perp


@pytest.fixture
def psiq() -> tuple[torch.Tensor, torch.Tensor]:
    """Stream function and potential vorticity."""
    ne, nl, nx, ny = 2, 3, 50, 75
    psi = torch.rand((ne, nl, nx + 1, ny + 1))
    q = torch.rand((ne, nl, nx, ny))
    return psi, q


def test_wide_advection(psiq: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test div_flux_5_pts_only."""
    imin = 10
    imax = 40
    jmin = 15
    jmax = 45

    psi, q = psiq
    u, v = grad_perp(psi)
    adv = div_flux_5pts(q, u[..., 1:-1, :], v[..., :, 1:-1], 1, 1)
    adv_slice = adv[..., imin:imax, jmin:jmax]

    psi_slice = psi[..., imin : imax + 1, jmin : jmax + 1]
    u_slice, v_slice = grad_perp(psi_slice)
    q_slice_wide = q[..., imin - 3 : imax + 3, jmin - 3 : jmax + 3]
    adv_ = div_flux_5pts_only(q_slice_wide, u_slice, v_slice, 1, 1)
    torch.testing.assert_close(adv_slice, adv_)
