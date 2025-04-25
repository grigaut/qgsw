"""Test UVH and UVHTAlpha."""

from __future__ import annotations

import pytest
import torch

from qgsw.exceptions import ParallelSlicingError
from qgsw.fields.variables.prognostic_tuples import (
    UVH,
    UVHT,
    BasePrognosticUVH,
    UVHTAlpha,
)
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
def uvht() -> UVHT:
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
    t = torch.rand(
        (n_ens,),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return UVHT(t=t, u=u, v=v, h=h)


@pytest.fixture
def uvht_alpha() -> UVHTAlpha:
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
    t = torch.rand(
        (n_ens,),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return UVHTAlpha(t=t, u=u, v=v, h=h, alpha=alpha)


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


def test_operations_uvht(
    uvht: UVHT,
) -> None:
    """Test operations on UVHT."""
    u0 = uvht.u
    v0 = uvht.v
    h0 = uvht.h
    t0 = uvht.t
    assert ((uvht * 2).u == 2 * u0).all()
    assert ((uvht * 2).v == 2 * v0).all()
    assert ((uvht * 2).h == 2 * h0).all()
    assert ((uvht * 2).t == t0).all()
    assert ((2 * uvht).u == 2 * u0).all()
    assert ((2 * uvht).v == 2 * v0).all()
    assert ((2 * uvht).h == 2 * h0).all()
    assert ((2 * uvht).t == t0).all()
    assert ((uvht + 1.2 * uvht).u == u0 + 1.2 * u0).all()
    assert ((uvht + 1.2 * uvht).v == v0 + 1.2 * v0).all()
    assert ((uvht + 1.2 * uvht).h == h0 + 1.2 * h0).all()
    assert ((uvht + 1.2 * uvht).t == t0).all()
    assert ((uvht - 3 * uvht).u == u0 - 3 * u0).all()
    assert ((uvht - 3 * uvht).v == v0 - 3 * v0).all()
    assert ((uvht - 3 * uvht).h == h0 - 3 * h0).all()
    assert ((uvht - 3 * uvht).t == t0).all()


def test_operations_uvht_alpha(
    uvht_alpha: UVHTAlpha,
) -> None:
    """Test operations on UVHTAlpha."""
    u0 = uvht_alpha.u
    v0 = uvht_alpha.v
    h0 = uvht_alpha.h
    alpha0 = uvht_alpha.alpha
    t0 = uvht_alpha.t
    assert ((uvht_alpha * 2).u == 2 * u0).all()
    assert ((uvht_alpha * 2).v == 2 * v0).all()
    assert ((uvht_alpha * 2).h == 2 * h0).all()
    assert ((uvht_alpha * 2).alpha == alpha0).all()
    assert ((uvht_alpha * 2).t == t0).all()
    assert ((2 * uvht_alpha).u == 2 * u0).all()
    assert ((2 * uvht_alpha).v == 2 * v0).all()
    assert ((2 * uvht_alpha).h == 2 * h0).all()
    assert ((2 * uvht_alpha).alpha == alpha0).all()
    assert ((2 * uvht_alpha).t == t0).all()
    assert ((uvht_alpha + 1.2 * uvht_alpha).u == u0 + 1.2 * u0).all()
    assert ((uvht_alpha + 1.2 * uvht_alpha).v == v0 + 1.2 * v0).all()
    assert ((uvht_alpha + 1.2 * uvht_alpha).h == h0 + 1.2 * h0).all()
    assert ((uvht_alpha + 1.2 * uvht_alpha).alpha == alpha0).all()
    assert ((uvht_alpha + 1.2 * uvht_alpha).t == t0).all()
    assert ((uvht_alpha - 3 * uvht_alpha).u == u0 - 3 * u0).all()
    assert ((uvht_alpha - 3 * uvht_alpha).v == v0 - 3 * v0).all()
    assert ((uvht_alpha - 3 * uvht_alpha).h == h0 - 3 * h0).all()
    assert ((uvht_alpha - 3 * uvht_alpha).alpha == alpha0).all()
    assert ((uvht_alpha - 3 * uvht_alpha).t == t0).all()


testdata = [
    pytest.param("uvh", id="uvh"),
    pytest.param("uvht", id="uvht"),
    pytest.param("uvht_alpha", id="uvhtalpha"),
]


@pytest.mark.parametrize(("prognostic"), testdata)
def test_slicing_uvh(
    prognostic: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test slicing on BasePrognosticUVH."""
    base_uvh: BasePrognosticUVH = request.getfixturevalue(prognostic)
    u, v, h = base_uvh.uvh.parallel_slice[:1, 0]
    u_ = base_uvh.uvh.u[:1, 0]
    v_ = base_uvh.uvh.v[:1, 0]
    h_ = base_uvh.uvh.h[:1, 0]
    torch.testing.assert_close(u, u_)
    torch.testing.assert_close(v, v_)
    torch.testing.assert_close(h, h_)


@pytest.mark.parametrize(("prognostic"), testdata)
def test_slicing_depth_uvh(
    prognostic: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test slicing depth error on BasePrognosticUVH."""
    base_uvh: BasePrognosticUVH = request.getfixturevalue(prognostic)
    with pytest.raises(ParallelSlicingError):
        base_uvh.uvh.parallel_slice[:1, 0, 0:1]
