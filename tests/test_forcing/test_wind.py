"""Wind forcing testing modules."""

import torch

from qgsw.forcing.wind import CosineZonalWindForcing
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit


def test_cosine_wind_forcing() -> None:
    """Test CosineZonalWindForcing."""
    lx = 2560.0e3
    nx = 128
    ly = 5120.0e3
    ny = 256

    x, y = torch.meshgrid(
        torch.linspace(
            0,
            lx,
            nx + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        torch.linspace(
            0,
            ly,
            ny + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        indexing="ij",
    )
    rho = 1000
    mag = 0.08  # Wind stress magnitude (Pa m-1 kg s-2)
    tau0 = mag / rho
    y_ugrid = 0.5 * (y[1:, :] + y[:-1, :])
    taux = tau0 * torch.cos(2 * torch.pi * (y_ugrid - ly / 2) / ly)

    space = SpaceDiscretization2D.from_tensors(
        x=torch.linspace(
            0,
            lx,
            nx + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        y=torch.linspace(
            0,
            ly,
            ny + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        x_unit=Unit.M,
        y_unit=Unit.M,
    )
    wf = CosineZonalWindForcing(
        space,
        mag,
        rho,
    )
    taux_, tauy_ = wf.compute()

    assert torch.isclose(taux, taux_).all()
    assert (tauy_ == 0).all()
