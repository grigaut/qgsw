"""Space Discretization Testing Module."""

import torch

from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.units._units import Unit


def test_omega_grid() -> None:
    """Verify that omega grid corresponds to the xy grid."""
    lx = 2560.0e3
    nx = 128
    ly = 5120.0e3
    ny = 256

    x, y = torch.meshgrid(
        torch.linspace(0, lx, nx + 1, dtype=torch.float64, device="cpu"),
        torch.linspace(0, ly, ny + 1, dtype=torch.float64, device="cpu"),
        indexing="ij",
    )
    space = SpaceDiscretization2D.from_tensors(
        x=torch.linspace(0, lx, nx + 1, dtype=torch.float64, device="cpu"),
        y=torch.linspace(0, ly, ny + 1, dtype=torch.float64, device="cpu"),
        x_unit=Unit.METERS,
        y_unit=Unit.METERS,
    )
    assert (space.omega.xy.x == x).all()
    assert (space.omega.xy.y == y).all()
