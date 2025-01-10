"""Test UVH and UVHalpha."""

from __future__ import annotations

import pytest
import torch

from qgsw.fields.variables.uvh import UVH, UVHalpha
from qgsw.specs import DEVICE


@pytest.fixture
def uvh() -> UVH:
    """UVH."""
    n_ens = 3
    nl = 2
    nx = 20
    ny = 30
    u = torch.rand(
        (n_ens, nl, nx + 1, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    v = torch.rand(
        (n_ens, nl, nx, ny + 1),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    h = torch.rand(
        (n_ens, nl, nx, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return UVH(u=u, v=v, h=h)


@pytest.fixture
def uvh_alpha() -> UVHalpha:
    """UVH alpha."""
    n_ens = 3
    nl = 2
    nx = 20
    ny = 30
    u = torch.rand(
        (n_ens, nl, nx + 1, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    v = torch.rand(
        (n_ens, nl, nx, ny + 1),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    h = torch.rand(
        (n_ens, nl, nx, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    alpha = torch.rand(
        (n_ens,),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return UVHalpha(u=u, v=v, h=h, alpha=alpha)


def test_operations_uvh(
    uvh: UVH,
) -> None:
    """Test operations on UVH."""
    u0 = uvh.u
    v0 = uvh.v
    h0 = uvh.h
    assert ((uvh * 2).u == 2 * u0).all()
    assert ((uvh * 2).v == 2 * v0).all()
    assert ((uvh * 2).h == 2 * h0).all()
    assert ((2 * uvh).u == 2 * u0).all()
    assert ((2 * uvh).v == 2 * v0).all()
    assert ((2 * uvh).h == 2 * h0).all()
    assert ((uvh + 1.2 * uvh).u == u0 + 1.2 * u0).all()
    assert ((uvh + 1.2 * uvh).v == v0 + 1.2 * v0).all()
    assert ((uvh + 1.2 * uvh).h == h0 + 1.2 * h0).all()
    assert ((uvh - 3 * uvh).u == u0 - 3 * u0).all()
    assert ((uvh - 3 * uvh).v == v0 - 3 * v0).all()
    assert ((uvh - 3 * uvh).h == h0 - 3 * h0).all()


def test_operations_uvh_alpha(
    uvh_alpha: UVHalpha,
) -> None:
    """Test operations on UVHalpha."""
    u0 = uvh_alpha.u
    v0 = uvh_alpha.v
    h0 = uvh_alpha.h
    alpha0 = uvh_alpha.alpha
    assert ((uvh_alpha * 2).u == 2 * u0).all()
    assert ((uvh_alpha * 2).v == 2 * v0).all()
    assert ((uvh_alpha * 2).h == 2 * h0).all()
    assert ((uvh_alpha * 2).alpha == alpha0).all()
    assert ((2 * uvh_alpha).u == 2 * u0).all()
    assert ((2 * uvh_alpha).v == 2 * v0).all()
    assert ((2 * uvh_alpha).h == 2 * h0).all()
    assert ((2 * uvh_alpha).alpha == alpha0).all()
    assert ((uvh_alpha + 1.2 * uvh_alpha).u == u0 + 1.2 * u0).all()
    assert ((uvh_alpha + 1.2 * uvh_alpha).v == v0 + 1.2 * v0).all()
    assert ((uvh_alpha + 1.2 * uvh_alpha).h == h0 + 1.2 * h0).all()
    assert ((uvh_alpha + 1.2 * uvh_alpha).alpha == alpha0).all()
    assert ((uvh_alpha - 3 * uvh_alpha).u == u0 - 3 * u0).all()
    assert ((uvh_alpha - 3 * uvh_alpha).v == v0 - 3 * v0).all()
    assert ((uvh_alpha - 3 * uvh_alpha).h == h0 - 3 * h0).all()
    assert ((uvh_alpha - 3 * uvh_alpha).alpha == alpha0).all()
