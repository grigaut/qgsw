"""Test state object."""

import torch

from qgsw.models.variables.core import UVH, State
from qgsw.models.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    SurfaceHeightAnomaly,
)


def test_init_update() -> None:
    """Test state initialization. and updates."""
    n_ens = 1
    nl = 2
    nx = 10
    ny = 10
    state = State.steady(n_ens, nl, nx, ny, dtype=torch.float64, device="cpu")

    assert state.u.shape == (n_ens, nl, nx + 1, ny)
    assert state.v.shape == (n_ens, nl, nx, ny + 1)
    assert state.h.shape == (n_ens, nl, nx, ny)

    assert (state.u == 0).all()
    assert (state.v == 0).all()
    assert (state.h == 0).all()

    u = torch.clone(state.u) + 1
    v = torch.clone(state.v) + 2
    h = torch.clone(state.h) + 3

    state.update(u, v, h)

    assert (state.u == u).all()
    assert (state.v == v).all()
    assert (state.h == h).all()

    u += 1
    v += 2
    h += 3

    state.uvh = UVH(u, v, h)

    assert (state.u == u).all()
    assert (state.v == v).all()
    assert (state.h == h).all()


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
    state.update(state.u, state.v, state.h + 2)
    # Assert all variables must be updated
    assert all(not var.up_to_date for var in state.diag_vars)
    # Compute the value of eta
    eta2 = eta_bound.get()
    # Assert eta has changed
    assert not (eta1 == eta2).all()
