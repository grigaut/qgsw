"""Tests for dynamic variables."""

import pytest
import torch

from qgsw.fields.variables.covariant import (
    MeridionalVelocityFlux,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.physical import (
    LayerDepthAnomaly,
    Pressure,
    SurfaceHeightAnomaly,
)
from qgsw.fields.variables.state import StateUVH
from qgsw.fields.variables.tuples import UVHT
from qgsw.specs import DEVICE


@pytest.fixture
def state() -> StateUVH:
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
    return StateUVH(UVHT(u, v, h, t))


def test_slicing(state: StateUVH) -> None:
    """Test slicing with variables."""
    # Variables
    h_phys = LayerDepthAnomaly()

    h_phys.slices = [slice(0, 1), slice(0, 1), ...]
    h = h_phys.compute(state.prognostic)
    assert h.shape == (1, 1, 10, 10)

    h_no_slice = h_phys.compute_no_slice(state.prognostic)
    assert h_no_slice.shape == (1, 2, 10, 10)

    assert (h_no_slice.__getitem__(h_phys.slices) == h).shape


def test_slicing_bound(state: StateUVH) -> None:
    """Test slicing with bounded variables."""
    # Variables
    eta_phys = SurfaceHeightAnomaly()
    p = Pressure(
        g_prime=torch.tensor(
            [[[[10]], [[0.05]]]],
            dtype=torch.float64,
            device=DEVICE.get(),
        ),
        eta_phys=eta_phys,
    )
    p_bound = p.bind(state)

    p_bound.slices = [slice(0, 1), slice(0, 1), ...]
    p_value = p_bound.get()
    assert p_value.shape == (1, 1, 10, 10)

    p_bound.slices = [...]
    p_no_slice = p_bound.get()
    assert p_no_slice.shape == (1, 2, 10, 10)

    assert (p_no_slice.__getitem__(p_bound.slices) == p_value).shape


def test_velocity_flux(state: StateUVH) -> None:
    """Test the velocity flux."""
    dx = 2
    dy = 3
    # Variables
    u_flux = ZonalVelocityFlux(dx=dx)
    v_flux = MeridionalVelocityFlux(dy=dy)
    # Compute momentum
    U = u_flux.compute(state.prognostic)
    V = v_flux.compute(state.prognostic)
    # Assert values equality
    assert (state.prognostic.u / dx**2 == U).all()
    assert (state.prognostic.v / dy**2 == V).all()
