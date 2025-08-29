"""Test inhomogeneous pv solver."""

import pytest
import torch

from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.pv_inversion import InhomogeneousPVInversion
from qgsw.specs import defaults


@pytest.fixture
def shape() -> tuple[int, int, int]:
    """Shape."""
    return (3, 25, 40)


@pytest.fixture
def pv(shape: tuple[int, int, int]) -> torch.Tensor:
    """Random potential vorticity."""
    return torch.rand(shape, **defaults.get())


@pytest.fixture
def boundaries(shape: tuple[int, int, int, int]) -> Boundaries:
    """Boundary conditions."""
    nl, nx, ny = shape
    x = torch.linspace(0, 2 * torch.pi, nx + 1, **defaults.get())
    y = torch.linspace(0, 2 * torch.pi, ny + 1, **defaults.get())

    top = torch.stack([torch.sin((i + 1) * x) for i in range(nl)])
    bottom = torch.stack([2 * torch.sin((i + 1) * x) for i in range(nl)])
    left = torch.stack([torch.sin((i + 1) * y) for i in range(nl)])
    right = torch.stack([2 * torch.sin((i + 1) * y) for i in range(nl)])
    return Boundaries(top, bottom, left, right)


def test_solver(pv: torch.Tensor, boundaries: Boundaries) -> None:
    """Test solver behavior by checking bioundaries."""
    A = torch.diag(  # noqa: N806
        torch.tensor([i + 1 for i in range(pv.shape[0])], **defaults.get())
    )
    solver = InhomogeneousPVInversion(A, 1, 1, 1)
    solver.set_boundaries(boundaries)
    sf = solver.compute_stream_function(pv)
    torch.testing.assert_close(sf[0, ..., :, -1], boundaries.top)
    torch.testing.assert_close(sf[0, ..., :, 0], boundaries.bottom)
    torch.testing.assert_close(sf[0, ..., 0, :], boundaries.left)
    torch.testing.assert_close(sf[0, ..., -1, :], boundaries.right)
