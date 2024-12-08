"""Tests for dynamic variables."""

import pytest
import torch

from qgsw.models.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalVelocity,
    VelocityFlux,
)
from qgsw.models.variables.state import UVH, State


@pytest.fixture
def state() -> State:
    """Instantiate a state variable."""
    # Shapes
    n_ens = 1
    nl = 2
    nx = 10
    ny = 10
    # Initialization
    u = torch.rand((n_ens, nl, nx + 1, ny), dtype=torch.float64, device="cpu")
    v = torch.rand((n_ens, nl, nx, ny + 1), dtype=torch.float64, device="cpu")
    h = torch.rand((n_ens, nl, nx, ny), dtype=torch.float64, device="cpu")
    return State(UVH(u, v, h))


def test_physical_prognostic_variables(state: State) -> None:
    """Test the physical progonstic variables."""
    dx = 2
    dy = 3
    # Variables
    uv_phys = PhysicalVelocity(dx=dx, dy=dy)
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)
    # Compute physical variables
    u_phys, v_phys = uv_phys.compute(state.uvh)
    h_phys = h_phys.compute(state.uvh)
    # Assert values equality
    assert (u_phys == state.uvh.u / dx).all()
    assert (v_phys == state.uvh.v / dy).all()
    assert (h_phys == state.uvh.h / (dx * dy)).all()


def test_velocity_flux(state: State) -> None:
    """Test the velocity flux."""
    dx = 2
    dy = 3
    # Variables
    momentum = VelocityFlux(dx=dx, dy=dy)
    # Compute momentum
    U, V = momentum.compute(state.uvh)  # noqa: N806
    # Assert values equality
    assert (state.uvh.u / dx**2 == U).all()
    assert (state.uvh.v / dy**2 == V).all()
