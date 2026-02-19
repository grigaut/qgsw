"""None perturbation Tests."""

import torch

from qgsw.perturbations.none import NoPerturbation
from qgsw.spatial.core.grid import Grid3D
from qgsw.specs import DEVICE


def test_no_perturbation() -> None:
    """Verify that NoPerturbation is null."""
    dx = 10
    dy = 10
    grid = Grid3D.from_coords(
        x_1d=torch.arange(
            0,
            100,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        * dx,
        y_1d=torch.arange(
            0,
            100,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        * dy,
        h_1d=torch.linspace(
            0,
            200,
            3,
            device=DEVICE.get(),
            dtype=torch.float64,
        ),
    )

    perturbation = NoPerturbation()
    p = perturbation.compute_initial_pressure(
        grid_3d=grid,
        f0=9.375e-5,
        Ro=0.1,
    )
    assert (p == 0).all()
