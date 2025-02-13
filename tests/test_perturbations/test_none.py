"""None perturbation Tests."""

import torch

from qgsw.perturbations.none import NoPerturbation
from qgsw.spatial.core.grid import Grid3D
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit


def test_no_perturbation() -> None:
    """Verify that NoPerturbation is null."""
    grid = Grid3D.from_tensors(
        x=torch.linspace(
            0,
            1000,
            100,
            device=DEVICE.get(),
            dtype=torch.float64,
        ),
        y=torch.linspace(
            0,
            1000,
            100,
            device=DEVICE.get(),
            dtype=torch.float64,
        ),
        h=torch.linspace(0, 200, 3, device=DEVICE.get(), dtype=torch.float64),
        x_unit=Unit.M,
        y_unit=Unit.M,
        zh_unit=Unit.M,
    )

    perturbation = NoPerturbation()
    p = perturbation.compute_initial_pressure(
        grid_3d=grid,
        f0=9.375e-5,
        Ro=0.1,
    )
    assert (p == 0).all()
