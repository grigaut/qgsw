"""Test boundary conditions."""

from collections.abc import Callable

import pytest
import torch

from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.boundary_conditions.interpolation import (
    BilinearExtendedBoundary,
)


@pytest.fixture
def field() -> torch.Tensor:
    """Random field."""
    return torch.rand((2, 5, 25, 40))


def test_extraction(field: torch.Tensor) -> None:
    """Test boundary extraction."""
    imin = 5
    imax = 20
    jmin = 10
    jmax = 30
    boundaries = Boundaries.extract(field, imin, imax, jmin, jmax)
    assert (boundaries.top == field[..., imin:imax, jmax - 1 : jmax]).all()
    assert (boundaries.bottom == field[..., imin:imax, jmin : jmin + 1]).all()
    assert (boundaries.left == field[..., imin : imin + 1, jmin:jmax]).all()
    assert (boundaries.right == field[..., imax - 1 : imax, jmin:jmax]).all()


def test_wide_extraction(field: torch.Tensor) -> None:
    """Test wide boundary extraction."""
    imin = 5
    imax = 20
    jmin = 10
    jmax = 30
    boundaries = Boundaries.extract(field, imin, imax, jmin, jmax, 2)
    assert (
        boundaries.top == field[..., imin - 1 : imax + 1, jmax - 1 : jmax + 1]
    ).all()
    assert (
        boundaries.bottom
        == field[..., imin - 1 : imax + 1, jmin - 1 : jmin + 1]
    ).all()
    assert (
        boundaries.left == field[..., imin - 1 : imin + 1, jmin - 1 : jmax + 1]
    ).all()
    assert (
        boundaries.right
        == field[..., imax - 1 : imax + 1, jmin - 1 : jmax + 1]
    ).all()


def test_boundary_interpolation(field: torch.Tensor) -> None:
    """Test boundary extraction."""
    imin = 5
    imax = 20
    jmin = 10
    jmax = 30
    boundaries = Boundaries.extract(field, imin, imax, jmin, jmax)
    be = BilinearExtendedBoundary(boundaries)
    boundary_interpolated = be.compute()
    top_i = boundary_interpolated[..., :, -1]
    bottom_i = boundary_interpolated[..., :, 0]
    left_i = boundary_interpolated[..., 0, :]
    right_i = boundary_interpolated[..., -1, :]
    torch.testing.assert_close(top_i, boundaries.top[..., 0])
    torch.testing.assert_close(bottom_i, boundaries.bottom[..., 0])
    torch.testing.assert_close(left_i, boundaries.left[..., 0, :])
    torch.testing.assert_close(right_i, boundaries.right[..., 0, :])


testoperations = [
    pytest.param(
        lambda f: f + f,
        id="add",
    ),
    pytest.param(
        lambda f: 2 * f,
        id="mul",
    ),
    pytest.param(
        lambda f: f * 2,
        id="rmul",
    ),
    pytest.param(
        lambda f: f / 2,
        id="truediv",
    ),
]


@pytest.mark.parametrize("operation", testoperations)
def test_operations(
    field: torch.Tensor,
    operation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Test operations on boundaries."""
    imin = 5
    imax = 20
    jmin = 10
    jmax = 30
    b = operation(Boundaries.extract(field, imin, imax, jmin, jmax))
    b2 = Boundaries.extract(operation(field), imin, imax, jmin, jmax)
    assert (b2.top == b.top).all()
    assert (b2.bottom == b.bottom).all()
    assert (b2.left == b.left).all()
    assert (b2.right == b.right).all()


def test_adding_boundary() -> None:
    """Test Boundary.set_to."""
    data = torch.rand((2, 3, 20, 30))
    imin = 5
    imax = 15
    jmin = 8
    jmax = 20
    data_int = data[..., imin + 2 : imax - 2, jmin + 2 : jmax - 2]
    boundary_inner = Boundaries.extract(
        data, imin + 1, imax - 1, jmin + 1, jmax - 1
    )
    boundary_outer = Boundaries.extract(data, imin, imax, jmin, jmax)
    data_ = torch.nn.functional.pad(data_int, (2, 2, 2, 2))
    boundary_inner.set_to(data_, offset=1, inplace=True)
    boundary_outer.set_to(data_, offset=0, inplace=True)
    torch.testing.assert_close(data_, data[..., imin:imax, jmin:jmax])


def test_expand() -> None:
    """Test Boundary.expand."""
    data = torch.rand((2, 3, 20, 30))
    imin = 5
    imax = 15
    jmin = 8
    jmax = 20
    data_ = data[..., imin + 2 : imax - 2, jmin + 2 : jmax - 2]
    boundary_inner = Boundaries.extract(
        data, imin + 1, imax - 1, jmin + 1, jmax - 1
    )
    boundary_outer = Boundaries.extract(data, imin, imax, jmin, jmax)
    data_ = boundary_inner.expand(data_)
    data_ = boundary_outer.expand(data_)
    torch.testing.assert_close(data_, data[..., imin:imax, jmin:jmax])


def test_wide_expansion() -> None:
    """Test wide boundary expansion."""
    data = torch.rand((2, 3, 20, 30))
    imin = 5
    imax = 15
    jmin = 8
    jmax = 20
    data_ = data[..., imin + 2 : imax - 2, jmin + 2 : jmax - 2]
    boundary = Boundaries.extract(
        data,
        imin + 1,
        imax - 1,
        jmin + 1,
        jmax - 1,
        width=2,
    )
    data_ = boundary.expand(data_)
    torch.testing.assert_close(data_, data[..., imin:imax, jmin:jmax])
