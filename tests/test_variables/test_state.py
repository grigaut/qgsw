"""Test state object."""

import torch

from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    PressureTilde,
)
from qgsw.fields.variables.state import State, StateAlpha
from qgsw.fields.variables.uvh import UVH
from qgsw.specs import DEVICE


def test_init_update() -> None:
    """Test state initialization. and updates."""
    n_ens = 1
    nl = 2
    nx = 10
    ny = 10
    state = State.steady(
        n_ens,
        nl,
        nx,
        ny,
        dtype=torch.float64,
        device=DEVICE.get(),
    )

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

    state.update_uvh(UVH(u, v, h))

    assert (state.u.get() == u).all()
    assert (state.v.get() == v).all()
    assert (state.h.get() == h).all()

    u += 1
    v += 2
    h += 3

    state.update_uvh(UVH(u, v, h))

    assert (state.u.get() == u).all()
    assert (state.v.get() == v).all()
    assert (state.h.get() == h).all()


def test_nested_bound_variables() -> None:
    """Verify the behavior of nested variables."""
    # Define state
    state = State.steady(1, 2, 10, 10, torch.float64, DEVICE.get())
    # Define variables
    h = PhysicalLayerDepthAnomaly(ds=1)
    eta_phys = PhysicalSurfaceHeightAnomaly(h_phys=h)
    # Bind only eta_phys
    eta_bound = eta_phys.bind(state)
    # Compute eta_phys and h
    eta0 = eta_phys.compute(state.prognostic)
    # Assert both variables are bound
    assert len(state.diag_vars) == 2  # noqa: PLR2004
    # Assert both variables are bound once
    assert len(state.diag_vars) == 2  # noqa: PLR2004
    # Compute eta_phys
    eta1 = eta_bound.get()
    # Compare values of eta_phys and h
    assert (eta0 == eta1).all()
    # Update state
    state.update_uvh(UVH(state.u.get(), state.v.get(), state.h.get() + 2))
    # Assert all variables must be updated
    assert all(not var.up_to_date for var in state.diag_vars.values())
    # Compute the value of eta_phys
    eta2 = eta_bound.get()
    # Assert eta_phys has changed
    assert not (eta1 == eta2).all()


def test_state_alpha_updates() -> None:
    """Test updates on StateAlpha."""
    state = StateAlpha.steady(
        1,
        2,
        10,
        10,
        torch.float64,
        DEVICE.get(),
    )
    alpha0 = state.alpha
    state.update_uvh(UVH(state.u.get() + 2, state.v.get(), state.h.get()))
    assert (state.alpha.get() == alpha0.get()).all()
    uvh0 = state.prognostic.uvh
    state.alpha = torch.tensor([0.4], dtype=torch.float64, device=DEVICE.get())
    assert state.prognostic.uvh == uvh0


def test_sate_update_only_alpha() -> None:
    """Test that updating only state does not reload all variables."""
    state = StateAlpha.steady(
        1,
        1,
        10,
        10,
        torch.float64,
        DEVICE.get(),
    )
    state.update_uvh(
        UVH(
            state.u.get(),
            state.v.get(),
            torch.rand_like(state.h.get()),
        ),
    )
    g_tilde = torch.tensor(
        [[[[9.81]], [[0.05]]]],
        dtype=torch.float64,
        device=DEVICE.get(),
    )
    h_phys = PhysicalLayerDepthAnomaly(ds=2).bind(state)
    h_tilde = PhysicalLayerDepthAnomaly(ds=2)
    eta_tilde = PhysicalSurfaceHeightAnomaly(h_tilde)
    pressure_tilde = PressureTilde(g_tilde, eta_tilde).bind(state)
    h_phys.get()
    pressure_tilde.get()
    assert h_phys.up_to_date
    assert pressure_tilde.up_to_date
    state.update_alpha(torch.rand_like(state.alpha.get()))
    assert h_phys.up_to_date
    assert not pressure_tilde.up_to_date
    pressure_tilde.get()
    state.update_uvh(
        UVH(
            torch.rand_like(state.u.get()),
            torch.rand_like(state.v.get()),
            torch.rand_like(state.h.get()),
        ),
    )
    assert not h_phys.up_to_date
    assert not pressure_tilde.up_to_date
