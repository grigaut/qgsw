"""Tests for dynamic variables."""

import pytest
import torch

from qgsw.fields.variables.dynamics import (
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
    Pressure,
    SurfaceHeightAnomaly,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import UVHT
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
    t = torch.rand(
        (n_ens,),
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    return State(UVHT(u, v, h, t))


def test_slicing(state: State) -> None:
    """Test slicing with variables."""
    dx = 2
    dy = 3
    # Variables
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)

    h_phys.slices = [slice(0, 1), slice(0, 1), ...]
    h = h_phys.compute(state.prognostic)
    assert h.shape == (1, 1, 10, 10)

    h_no_slice = h_phys.compute_no_slice(state.prognostic)
    assert h_no_slice.shape == (1, 2, 10, 10)

    assert (h_no_slice.__getitem__(h_phys.slices) == h).shape


def test_slicing_bound(state: State) -> None:
    """Test slicing with bounded variables."""
    dx = 2
    dy = 3
    # Variables
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)
    eta = SurfaceHeightAnomaly(h_phys)
    p = Pressure(
        g_prime=torch.tensor(
            [[[[10]], [[0.05]]]],
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        eta=eta,
    )
    p_bound = p.bind(state)

    p_bound.slices = [slice(0, 1), slice(0, 1), ...]
    p_value = p_bound.get()
    assert p_value.shape == (1, 1, 10, 10)

    p_bound.slices = [...]
    p_no_slice = p_bound.get()
    assert p_no_slice.shape == (1, 2, 10, 10)

    assert (p_no_slice.__getitem__(p_bound.slices) == p_value).shape


def test_physical_prognostic_variables(state: State) -> None:
    """Test the physical prognostic variables."""
    dx = 2
    dy = 3
    # Variables
    u_phys = PhysicalZonalVelocity(dx=dx)
    v_phys = PhysicalMeridionalVelocity(dy=dy)
    h_phys = PhysicalLayerDepthAnomaly(ds=dx * dy)
    # Compute physical variables
    u_phys = u_phys.compute(state.prognostic)
    v_phys = v_phys.compute(state.prognostic)
    h_phys = h_phys.compute(state.prognostic)
    # Assert values equality
    assert (u_phys == state.prognostic.u / dx).all()
    assert (v_phys == state.prognostic.v / dy).all()
    assert (h_phys == state.prognostic.h / (dx * dy)).all()


def test_velocity_flux(state: State) -> None:
    """Test the velocity flux."""
    dx = 2
    dy = 3
    # Variables
    u_flux = ZonalVelocityFlux(dx=dx)
    v_flux = MeridionalVelocityFlux(dy=dy)
    # Compute momentum
    U = u_flux.compute(state.prognostic)  # noqa: N806
    V = v_flux.compute(state.prognostic)  # noqa: N806
    # Assert values equality
    assert (state.prognostic.u / dx**2 == U).all()
    assert (state.prognostic.v / dy**2 == V).all()
