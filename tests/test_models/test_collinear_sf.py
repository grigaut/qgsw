"""Test collinear QG model."""

import pytest
import torch

from qgsw.models.qg.collinear_sublayer.core import QGCollinearSF
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.units._units import Unit
from qgsw.specs import DEVICE


@pytest.fixture
def space_2d() -> SpaceDiscretization2D:
    """X,Y and H."""
    lx = 2560.0e3
    nx = 128
    ly = 5120.0e3
    ny = 256

    X = torch.linspace(0, lx, nx + 1, dtype=torch.float64, device=DEVICE.get())  # noqa: N806
    Y = torch.linspace(0, ly, ny + 1, dtype=torch.float64, device=DEVICE.get())  # noqa:N806
    return SpaceDiscretization2D.from_tensors(
        x=X,
        y=Y,
        x_unit=Unit.M,
        y_unit=Unit.M,
    )


@pytest.fixture
def H() -> torch.Tensor:  # noqa: N802
    """Reference layer depth."""
    return torch.tensor([200, 800], dtype=torch.float64, device=DEVICE.get())


@pytest.fixture
def g_prime() -> torch.Tensor:
    """Reduced gravity."""
    return torch.tensor([10, 0.05], dtype=torch.float64, device=DEVICE.get())


def test_H_shape(  # noqa: N802
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> None:
    """Check H shape."""
    model = QGCollinearSF(
        space_2d,
        H,
        g_prime,
    )
    assert model.H.shape == (1, 1, 1)


def test_g_prime_shape(
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> None:
    """Check g' shape."""
    model = QGCollinearSF(
        space_2d,
        H,
        g_prime,
    )
    assert model.g_prime.shape == (1, 1, 1)


def test_UVH_shape(  # noqa: N802
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> None:
    """Check UVH shape."""
    model = QGCollinearSF(
        space_2d,
        H,
        g_prime,
    )
    assert model.u.shape[1] == 1
    assert model.v.shape[1] == 1
    assert model.h.shape[1] == 1


def test_stretching_matrix_shape(
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> None:
    """Check A shape."""
    model = QGCollinearSF(
        space_2d,
        H,
        g_prime,
    )
    model.coefficient = 0
    assert model.A.shape == (1, 1)


testdata = [
    pytest.param(
        0,
        torch.tensor(
            [[1 / 200 / 10 + 1 / 200 / 0.05]],
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        id="alpha=0",
    ),
    pytest.param(
        1,
        torch.tensor(
            [[1 / 200 / 10 + 1 / 200 / 0.05 - 1 / 200 / 0.05]],
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        id="alpha=1",
    ),
    pytest.param(
        0.5,
        torch.tensor(
            [[1 / 200 / 10 + 1 / 200 / 0.05 - 0.5 / 200 / 0.05]],
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        id="alpha=0.5",
    ),
]


@pytest.mark.parametrize(("coefficient", "reference"), testdata)
def test_model_stretching_matrix(
    space_2d: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
    coefficient: float,
    reference: torch.Tensor,
) -> None:
    """Test streching matrix computation from QG model."""
    model = QGCollinearSF(
        space_2d,
        H,
        g_prime,
    )
    model.coefficient = coefficient
    assert torch.isclose(reference, model.A).all()
