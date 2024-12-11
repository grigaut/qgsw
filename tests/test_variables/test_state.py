"""Test state object."""

import torch

from qgsw.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    SurfaceHeightAnomaly,
)
from qgsw.variables.state import State
from qgsw.variables.uvh import UVH


def test_init_update() -> None:
    """Test state initialization. and updates."""
    n_ens = 1
    nl = 2
    nx = 10
    ny = 10
    state = State.steady(n_ens, nl, nx, ny, dtype=torch.float64, device="cpu")

    u = state.u.get()
    v = state.v.get()
    h = state.h.get()

    assert u.shape == (n_ens, nl, nx + 1, ny)
    assert v.shape == (n_ens, nl, nx, ny + 1)
    assert h.shape == (n_ens, nl, nx, ny)

    assert (u == 0).all()
    assert (v == 0).all()
    assert (h == 0).all()

    u = torch.clone(u) + 1
    v = torch.clone(v) + 2
    h = torch.clone(h) + 3

    state.update(u, v, h)

    assert (state.u.get() == u).all()
    assert (state.v.get() == v).all()
    assert (state.h.get() == h).all()

    u += 1
    v += 2
    h += 3

    state.uvh = UVH(u, v, h)

    assert (state.u.get() == u).all()
    assert (state.v.get() == v).all()
    assert (state.h.get() == h).all()


def test_nested_bound_variables() -> None:
    """Verify the behavior of nested variables."""
    # Define state
    state = State.steady(1, 2, 10, 10, torch.float64, "cpu")
    # Define variables
    h = PhysicalLayerDepthAnomaly(ds=1)
    eta = SurfaceHeightAnomaly(h_phys=h)
    # Bind only eta
    eta_bound = eta.bind(state)
    # Compute eta and h
    eta0 = eta.compute(state.uvh)
    # Assert both variables are bound
    assert len(state.diag_vars) == 2  # noqa: PLR2004
    # Assert both variables are bound once
    assert len(state.diag_vars) == 2  # noqa: PLR2004
    # Compute eta
    eta1 = eta_bound.get()
    # Compare values of eta and h
    assert (eta0 == eta1).all()
    # Update state
    state.update(state.u.get(), state.v.get(), state.h.get() + 2)
    # Assert all variables must be updated
    assert all(not var.up_to_date for var in state.diag_vars)
    # Compute the value of eta
    eta2 = eta_bound.get()
    # Assert eta has changed
    assert not (eta1 == eta2).all()
