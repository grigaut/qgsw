"""Rescaling tests."""

from typing import TYPE_CHECKING

import pytest
import torch

from qgsw.exceptions import RescalingShapeMismatchError
from qgsw.fields.variables.prognostic_tuples import UVH
from qgsw.models.synchronization.rescaling import Rescaler
from qgsw.specs import defaults

if TYPE_CHECKING:
    from qgsw.models.base import ModelUVH


@pytest.fixture
def rescaler() -> Rescaler:
    """Rescaler fixture."""
    return Rescaler(nx_out=100, ny_out=200, dx_out=10, dy_out=10)


def uvh_nxny(nx: int, ny: int, n_ens: int = 1, nl: int = 2) -> UVH:
    """Create random UVH from shapes.

    Args:
        nx (int): Number of points in the X direction.
        ny (int): Number of points in the Y direction.
        n_ens (int, optional): Number of ensembles. Defaults to 1.
        nl (int, optional): Number of layers. Defaults to 2.

    Returns:
        UVH: UVH.
    """
    return UVH(
        torch.rand((n_ens, nl, nx + 1, ny), **defaults.get()),
        torch.rand((n_ens, nl, nx, ny + 1), **defaults.get()),
        torch.rand((n_ens, nl, nx, ny), **defaults.get()),
    )


testdata = [
    pytest.param(50, 100, id="small-to-large"),
    pytest.param(200, 400, id="large-to-small"),
]


@pytest.mark.parametrize(("nxin", "nyin"), testdata)
def test_rescaling(rescaler: Rescaler, nxin: int, nyin: int) -> None:
    """Test rescaling shapes validity."""
    uvh = uvh_nxny(nxin, nyin)
    uvh_i = rescaler(uvh)
    assert uvh_i.h.shape[-2:] == rescaler.output_hshape


testdata = [
    pytest.param(51, 50, id="small-to-large"),
    pytest.param(200, 201, id="large-to-small"),
]


@pytest.mark.parametrize(("nxin", "nyin"), testdata)
def test_rescaling_shape_mismatch(
    rescaler: Rescaler,
    nxin: int,
    nyin: int,
) -> None:
    """Test rescaling error when input shapes are not matching."""
    uvh = uvh_nxny(nxin, nyin)
    with pytest.raises(RescalingShapeMismatchError):
        rescaler(uvh)


testdata = [
    pytest.param("QG_model", id="QG"),
    pytest.param("SW_model", id="SW"),
]


@pytest.mark.parametrize(("model_fixture"), testdata)
def test_from_model(
    model_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test `for_model` method."""
    model: ModelUVH = request.getfixturevalue(model_fixture)
    rescaler = Rescaler.for_model(model)
    assert rescaler.output_hshape == (model.space.nx, model.space.ny)


testdata = [
    pytest.param("QG_model", 50, 100, id="small-to-large-QG"),
    pytest.param("SW_model", 50, 100, id="small-to-large-SW"),
    pytest.param("QG_model", 200, 400, id="large-to-small-QG"),
    pytest.param("SW_model", 200, 400, id="large-to-small-SW"),
]


@pytest.mark.parametrize(("model_fixture", "nxin", "nyin"), testdata)
def test_masks(
    model_fixture: str,
    rescaler: Rescaler,
    nxin: int,
    nyin: int,
    request: pytest.FixtureRequest,
) -> None:
    """Test rescaling error when input shapes are not matching."""
    model: ModelUVH = request.getfixturevalue(model_fixture)
    rescaler = Rescaler.for_model(model)
    uvh = uvh_nxny(nxin, nyin)
    uvh_i = rescaler(uvh)
    assert (uvh_i.u * model.masks.u == uvh_i.u).all()
    assert (uvh_i.v * model.masks.v == uvh_i.v).all()
    assert (uvh_i.h * model.masks.h == uvh_i.h).all()
