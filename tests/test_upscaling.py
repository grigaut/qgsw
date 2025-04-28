"""Test for upscaling objects."""

import pytest
import torch

from qgsw.fields.variables.prognostic_tuples import UVH, UVHT, UVHTAlpha
from qgsw.specs import defaults
from qgsw.upscaling import Upscaler


@pytest.fixture
def uvh() -> UVH:
    """UVH."""
    u = torch.rand((2, 3, 25 + 1, 30), **defaults.get())
    v = torch.rand((2, 3, 25, 30 + 1), **defaults.get())
    h = torch.rand((2, 3, 25, 30), **defaults.get())
    return UVH(u=u, v=v, h=h)


@pytest.fixture
def uvht(uvh: UVH) -> UVHT:
    """UVHT."""
    t = torch.rand((uvh.h.shape[:1]), **defaults.get())
    return UVHT.from_uvh(t=t, uvh=uvh)


@pytest.fixture
def uvht_alpha(uvh: UVH) -> UVHTAlpha:
    """UVHTAlpha."""
    t = torch.rand((uvh.h.shape[:1]), **defaults.get())
    alpha = torch.rand_like(uvh.h)
    return UVHTAlpha.from_uvh(t=t, alpha=alpha, uvh=uvh)


testdata = [
    pytest.param("uvh", id="uvh"),
    pytest.param("uvht", id="uvht"),
    pytest.param("uvht_alpha", id="uvht-alpha"),
]


@pytest.mark.parametrize("prognostic_str", testdata)
def test_upscaling(
    prognostic_str: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test upscaling.

    Args:
        prognostic_str (str): Prognostic tuple fixture name.
        request (pytest.FixtureRequest): Fixture request.
    """
    upscaler = Upscaler()
    delta = 2
    prognostic: UVH | UVHT | UVHTAlpha = request.getfixturevalue(
        prognostic_str,
    )
    prognostic_up: UVH | UVHT | UVHTAlpha = upscaler(prognostic, delta)
    n_ens, nl, nx, ny = prognostic.h.shape
    assert prognostic_up.u.shape == (
        n_ens,
        nl,
        delta * nx + 1,
        delta * ny,
    )
    assert (prognostic_up.u == 0).all()
    assert prognostic_up.v.shape == (
        n_ens,
        nl,
        delta * nx,
        delta * ny + 1,
    )
    assert (prognostic_up.v == 0).all()
    assert prognostic_up.h.shape == (n_ens, nl, delta * nx, delta * ny)
