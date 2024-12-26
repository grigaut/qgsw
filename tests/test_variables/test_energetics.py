"""Energetics functions tests."""

import pytest
import torch

from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    Pressure,
    StreamFunction,
    SurfaceHeightAnomaly,
)
from qgsw.fields.variables.energetics import (
    ModalAvailablePotentialEnergy,
    ModalKineticEnergy,
    TotalAvailablePotentialEnergy,
    TotalKineticEnergy,
    compute_W,
)
from qgsw.fields.variables.uvh import UVH
from qgsw.models.qg.stretching_matrix import compute_A


@pytest.fixture
def dx() -> float:
    """Dx."""
    return 2


@pytest.fixture
def dy() -> float:
    """Dy."""
    return 3


@pytest.fixture
def f0() -> float:
    """Coriolis parameter."""
    return 9.375e-5


@pytest.fixture
def H() -> torch.Tensor:  # noqa: N802
    """Reference layer depth."""
    h1 = 200
    h2 = 800

    return torch.tensor([h1, h2], dtype=torch.float64, device="cpu")


@pytest.fixture
def g_prime() -> torch.Tensor:
    """Reduced gravity."""
    g1 = 10
    g2 = 0.05
    return torch.tensor([g1, g2], dtype=torch.float64, device="cpu")


@pytest.fixture
def A(H: torch.Tensor, g_prime: torch.Tensor) -> torch.Tensor:  # noqa: N802, N803
    """Strecthing matrix."""
    return compute_A(H, g_prime, dtype=torch.float64, device="cpu")


@pytest.fixture
def psi(
    dx: float,
    dy: float,
    g_prime: torch.Tensor,
    f0: float,
) -> StreamFunction:
    """Stream function."""
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)
    eta = SurfaceHeightAnomaly(h_phys)
    p = Pressure(g_prime.unsqueeze(1).unsqueeze(1), eta)
    return StreamFunction(p, f0)


def test_energy_equality(
    psi: StreamFunction,
    A: torch.Tensor,  # noqa: N803
    H: torch.Tensor,  # noqa: N803
    dx: float,
    dy: float,
    f0: float,
) -> None:
    """Test equality between modal and layer-wise energy total."""
    ne = 3
    nl = 2
    nx = 30
    ny = 20

    ke_var = TotalKineticEnergy(psi, H, dx, dy)
    ke_hat_var = ModalKineticEnergy(A, psi, H, dx, dy)
    ape_var = TotalAvailablePotentialEnergy(A, psi, H, f0)
    ape_hat_var = ModalAvailablePotentialEnergy(A, psi, H, f0)

    uvh = UVH(
        torch.rand((ne, nl, nx + 1, ny), dtype=torch.float64, device="cpu"),
        torch.rand((ne, nl, nx, ny + 1), dtype=torch.float64, device="cpu"),
        torch.rand((ne, nl, nx, ny), dtype=torch.float64, device="cpu"),
    )
    ke = ke_var.compute(uvh)
    ke_hat = torch.sum(ke_hat_var.compute(uvh), dim=-1)
    assert torch.isclose(ke, ke_hat, rtol=1e-13, atol=0).all()
    ape = ape_var.compute(uvh)
    ape_hat = torch.sum(ape_hat_var.compute(uvh), dim=-1)
    assert torch.isclose(ape, ape_hat, rtol=1e-13, atol=0).all()


@pytest.mark.parametrize(
    ("H"),
    [
        (torch.tensor([100], dtype=torch.float64, device="cpu")),
        (torch.tensor([200, 800], dtype=torch.float64, device="cpu")),
    ],
)
def test_W_shape(H: torch.Tensor) -> None:  # noqa: N802, N803
    """Test W shape."""
    W = compute_W(H)  # noqa: N806
    assert W.shape == (H.shape[0], H.shape[0])
    assert torch.diag(W).shape == H.shape
