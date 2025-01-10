"""Test UVH and UVHalpha."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from qgsw.fields.variables.uvh import UVH, UVHalpha
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from collections.abc import Callable


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
def uvhalpha() -> UVHalpha:
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


testdata = [
    pytest.param(
        "uvh",
        lambda uvh: uvh.u,
        id="UVH-u",
    ),
    pytest.param(
        "uvh",
        lambda uvh: uvh.v,
        id="UVH-v",
    ),
    pytest.param(
        "uvh",
        lambda uvh: uvh.h,
        id="UVH-h",
    ),
    pytest.param(
        "uvhalpha",
        lambda uvhalpha: uvhalpha.u,
        id="UVHalpha-u",
    ),
    pytest.param(
        "uvhalpha",
        lambda uvhalpha: uvhalpha.v,
        id="UVHalpha-v",
    ),
    pytest.param(
        "uvhalpha",
        lambda uvhalpha: uvhalpha.h,
        id="UVHalpha-h",
    ),
    pytest.param(
        "uvhalpha",
        lambda uvhalpha: uvhalpha.alpha,
        id="UVHalpha-alpha",
    ),
]


@pytest.mark.parametrize(
    ("var_fixture", "var_get_method"),
    testdata,
)
def test_operations(
    var_fixture: str,
    var_get_method: Callable[[UVH | UVHalpha], torch.Tensor],
    request: pytest.FixtureRequest,
) -> None:
    """Test operations on UVH and UVHalpha."""
    var = request.getfixturevalue(var_fixture)
    value0 = var_get_method(var)
    assert (var_get_method(var * 2) == 2 * value0).all()
    assert (var_get_method(3 * var) == 3 * value0).all()
    assert (var_get_method(var + var) == value0 + value0).all()
    assert (var_get_method(var - var) == value0 - value0).all()
