"""Space Discretization Testing Module."""

import pytest
import torch

from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit


@pytest.fixture
def X_Y_H() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: N802
    """X,Y and H."""
    lx = 2560.0e3
    nx = 128
    ly = 5120.0e3
    ny = 256

    X = torch.linspace(0, lx, nx + 1, dtype=torch.float64, device=DEVICE.get())
    Y = torch.linspace(0, ly, ny + 1, dtype=torch.float64, device=DEVICE.get())
    H = torch.tensor([1, 3, 1], dtype=torch.float64, device=DEVICE.get())

    return X, Y, H


def test_omega_grid(
    X_Y_H: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Verify that omega grid corresponds to the xy grid."""
    X, Y, _ = X_Y_H
    x, y = torch.meshgrid(X, Y, indexing="ij")
    space = SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    assert (space.omega.xy.x == x).all()
    assert (space.omega.xy.y == y).all()


def test_2D_to_3D(  # noqa: N802
    X_Y_H: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test 2D to 3D space conversion."""
    X, Y, H = X_Y_H
    space_2d = SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    space_ref = SpaceDiscretization3D.from_tensors(
        x_unit=Unit.M,
        y_unit=Unit.M,
        zh_unit=Unit.M,
        x=X,
        y=Y,
        h=H,
    )
    space_3d = space_2d.add_h(Coordinates1D(points=H, unit=Unit.M))

    assert (space_3d.omega.xyh.x == space_ref.omega.xyh.x).all()
    assert (space_3d.omega.xyh.y == space_ref.omega.xyh.y).all()
    assert (space_3d.omega.xyh.h == space_ref.omega.xyh.h).all()
    assert (space_3d.u.xyh.x == space_ref.u.xyh.x).all()
    assert (space_3d.u.xyh.y == space_ref.u.xyh.y).all()
    assert (space_3d.u.xyh.h == space_ref.u.xyh.h).all()
    assert (space_3d.v.xyh.x == space_ref.v.xyh.x).all()
    assert (space_3d.v.xyh.y == space_ref.v.xyh.y).all()
    assert (space_3d.v.xyh.h == space_ref.v.xyh.h).all()
    assert (space_3d.h.xyh.x == space_ref.h.xyh.x).all()
    assert (space_3d.h.xyh.y == space_ref.h.xyh.y).all()
    assert (space_3d.h.xyh.h == space_ref.h.xyh.h).all()


def test_2D_and_3D_horizontal(  # noqa: N802
    X_Y_H: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test 2D to 3D space conversion."""
    X, Y, H = X_Y_H
    space_2d = SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    space_ref = SpaceDiscretization3D.from_tensors(
        x_unit=Unit.M,
        y_unit=Unit.M,
        zh_unit=Unit.M,
        x=X,
        y=Y,
        h=H,
    )
    assert (space_2d.omega.xy.x == space_ref.omega.xyh.x).all()
    assert (space_2d.omega.xy.y == space_ref.omega.xyh.y).all()
    assert (space_2d.u.xy.x == space_ref.u.xyh.x).all()
    assert (space_2d.u.xy.y == space_ref.u.xyh.y).all()
    assert (space_2d.v.xy.x == space_ref.v.xyh.x).all()
    assert (space_2d.v.xy.y == space_ref.v.xyh.y).all()
    assert (space_2d.h.xy.x == space_ref.h.xyh.x).all()
    assert (space_2d.h.xy.y == space_ref.h.xyh.y).all()


def test_slicing(
    X_Y_H: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Test 2D space slicing."""
    X, Y, _ = X_Y_H
    space_2d = SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    imin, imax = 64, 128
    jmin, jmax = 32, 132
    space_sliced = space_2d.slice(imin, imax + 1, jmin, jmax + 1)
    space_ref = SpaceDiscretization2D.from_tensors(
        x=space_2d.omega.xy.x[imin : imax + 1, 0],
        y=space_2d.omega.xy.y[0, jmin : jmax + 1],
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    assert (space_sliced.omega.xy.x == space_ref.omega.xy.x).all()
    assert (space_sliced.omega.xy.y == space_ref.omega.xy.y).all()
    assert (space_sliced.u.xy.x == space_ref.u.xy.x).all()
    assert (space_sliced.u.xy.y == space_ref.u.xy.y).all()
    assert (space_sliced.v.xy.x == space_ref.v.xy.x).all()
    assert (space_sliced.v.xy.y == space_ref.v.xy.y).all()
    assert (space_sliced.h.xy.x == space_ref.h.xy.x).all()
    assert (space_sliced.h.xy.y == space_ref.h.xy.y).all()
