"""Useful fixtures."""

import pytest
import torch

from qgsw.models.qg.uvh.core import QG
from qgsw.models.sw.core import SW
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import defaults
from qgsw.utils.units._units import Unit


@pytest.fixture
def H() -> torch.Tensor:  # noqa: N802
    """H."""
    return torch.tensor([400, 1100], **defaults.get())


@pytest.fixture
def g_prime() -> torch.Tensor:
    """Reduced gravity."""
    return torch.tensor([9.81, 0.025], **defaults.get())


@pytest.fixture
def space_2d() -> SpaceDiscretization2D:
    """Space 2D."""
    lx = 2560.0e3
    nx = 50
    ly = 5120.0e3
    ny = 100

    X = torch.linspace(0, lx, nx + 1, **defaults.get())  # noqa:N806
    Y = torch.linspace(0, ly, ny + 1, **defaults.get())  # noqa:N806
    return SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )


@pytest.fixture
def QG_model(  # noqa: N802
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> QG:
    """QG model fixture."""
    model = QG(
        space_2d=space_2d,
        H=H,
        g_prime=g_prime,
        beta_plane=BetaPlane(f0=9.375e-5, beta=1.754e-11),
        optimize=True,
    )
    model.bottom_drag_coef = 3.60577e-8
    model.slip_coef = 1.0
    model.dt = 300
    model.name = "Test-QG"
    return model


@pytest.fixture
def SW_model(  # noqa: N802
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> SW:
    """SW model fixture."""
    model = SW(
        space_2d=space_2d,
        H=H,
        g_prime=g_prime,
        beta_plane=BetaPlane(f0=9.375e-5, beta=1.754e-11),
        optimize=True,
    )
    model.bottom_drag_coef = 3.60577e-8
    model.slip_coef = 1.0
    model.dt = 300
    model.name = "Test-SW"
    return model
