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
    assert (boundaries.top == field[..., imin : imax + 1, jmax]).all()
    assert (boundaries.bottom == field[..., imin : imax + 1, jmin]).all()
    assert (boundaries.left == field[..., imin, jmin : jmax + 1]).all()
    assert (boundaries.right == field[..., imax, jmin : jmax + 1]).all()


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
    torch.testing.assert_close(top_i, boundaries.top)
    torch.testing.assert_close(bottom_i, boundaries.bottom)
    torch.testing.assert_close(left_i, boundaries.left)
    torch.testing.assert_close(right_i, boundaries.right)


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
    field: torch.Tensor, operation: Callable[[torch.Tensor], torch.Tensor]
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
