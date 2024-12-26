"""Tests for dynamic variables."""

import pytest
import torch

from qgsw.fields.variables.dynamics import (
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.state import UVH, State
from qgsw.specs import DEVICE


@pytest.fixture
def state() -> State:
    """Instantiate a state variable."""
    # Shapes
    n_ens = 1
    nl = 2
    nx = 10
    ny = 10
    # Initialization
    u = torch.rand(
        (n_ens, nl, nx + 1, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    v = torch.rand(
        (n_ens, nl, nx, ny + 1),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    h = torch.rand(
        (n_ens, nl, nx, ny),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return State(UVH(u, v, h))


def test_physical_prognostic_variables(state: State) -> None:
    """Test the physical progonstic variables."""
    dx = 2
    dy = 3
    # Variables
    u_phys = PhysicalZonalVelocity(dx=dx)
    v_phys = PhysicalMeridionalVelocity(dy=dy)
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)
    # Compute physical variables
    u_phys = u_phys.compute(state.uvh)
    v_phys = v_phys.compute(state.uvh)
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
    u_flux = ZonalVelocityFlux(dx=dx)
    v_flux = MeridionalVelocityFlux(dy=dy)
    # Compute momentum
    U = u_flux.compute(state.uvh)  # noqa: N806
    V = v_flux.compute(state.uvh)  # noqa: N806
    # Assert values equality
    assert (state.uvh.u / dx**2 == U).all()
    assert (state.uvh.v / dy**2 == V).all()
