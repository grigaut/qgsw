"""Shallow-water implementation.

Louis Thiry, Nov 2023 for IFREMER.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.models.base import Model
from qgsw.models.core import finite_diff
from qgsw.models.core.helmholtz import HelmholtzNeumannSolver
from qgsw.models.core.helmholtz_multigrid import MG_Helmholtz


def reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pytorch cumsum in the reverse order.

    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])
    """
    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


def inv_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of reverse cumsum function.

    Args:
        x (torch.Tensor): Tensor to perform inverse reverse cumsum on.
        dim (int): Dimension to perform the inverse reverse cumsum on.

    Returns:
        torch.Tensor: Inverse Reverse Cumsum
    """
    return torch.cat([-torch.diff(x, dim=dim), x.narrow(dim, -1, 1)], dim=dim)


class SW(Model):
    """# Implementation of multilayer rotating shallow-water model.

    Following https://doi.org/10.1029/2021MS002663 .

    ## Main ingredients
        - vector invariant formulation
        - velocity RHS using vortex force upwinding with wenoz-5 reconstruction
        - mass continuity RHS with finite volume using wenoz-5 recontruction

    ## Variables
    Prognostic variables u, v, h differ from physical variables
    u_phys, v_phys (velocity components) and
    h_phys (layer thickness perturbation) as they include
    metric terms dx and dy :
      - u = u_phys x dx
      - v = v_phys x dy
      - h = g_phys x dx x dy

    Diagnostic variables are :
      - U = u_phys / dx
      - V = v_phys / dx
      - omega = omega_phys x dx x dy    (rel. vorticity)
      - eta = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """

    def __init__(self, param: dict[str, Any]) -> None:
        """Parameters

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny),
            unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing,
            x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing,
            y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip,
            inbetween for
                        partial free slip.
            'bottom_drag_coef': float, linear bottom drag coefficient
        """
        super().__init__(param)

    def set_physical_uvh(
        self,
        u_phys: torch.Tensor | np.ndarray,
        v_phys: torch.Tensor | np.ndarray,
        h_phys: torch.Tensor | np.ndarray,
    ) -> None:
        """Set state variables from physical variables.

        Args:
            u_phys (torch.Tensor|np.ndarray): Physical U.
            v_phys (torch.Tensor|np.ndarray): Physical V.
            h_phys (torch.Tensor|np.ndarray): Physical H.
        """
        u_ = (
            torch.from_numpy(u_phys)
            if isinstance(u_phys, np.ndarray)
            else u_phys
        )
        v_ = (
            torch.from_numpy(v_phys)
            if isinstance(v_phys, np.ndarray)
            else v_phys
        )
        h_ = (
            torch.from_numpy(h_phys)
            if isinstance(h_phys, np.ndarray)
            else h_phys
        )
        u_ = u_.to(self.device)
        v_ = u_.to(self.device)
        h_ = u_.to(self.device)
        assert u_ * self.masks.u == u_, (
            "Input velocity u incoherent with domain mask, "
            "velocity must be zero out of domain."
        )
        assert v_ * self.masks.v == v_, (
            "Input velocity v incoherent with domain mask, "
            "velocity must be zero out of domain."
        )
        self.u = u_.type(self.dtype) * self.masks.u * self.dx
        self.v = v_.type(self.dtype) * self.masks.v * self.dy
        self.h = h_.type(self.dtype) * self.masks.h * self.area
        self.compute_diagnostic_variables()

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        dt0_u, dt0_v, dt0_h = self.compute_time_derivatives()
        self.u += self.dt * dt0_u
        self.v += self.dt * dt0_v
        self.h += self.dt * dt0_h

        dt1_u, dt1_v, dt1_h = self.compute_time_derivatives()
        self.u += (self.dt / 4) * (dt1_u - 3 * dt0_u)
        self.v += (self.dt / 4) * (dt1_v - 3 * dt0_v)
        self.h += (self.dt / 4) * (dt1_h - 3 * dt0_h)

        dt2_u, dt2_v, dt2_h = self.compute_time_derivatives()
        self.u += (self.dt / 12) * (8 * dt2_u - dt1_u - dt0_u)
        self.v += (self.dt / 12) * (8 * dt2_v - dt1_v - dt0_v)
        self.h += (self.dt / 12) * (8 * dt2_h - dt1_h - dt0_h)

    def advection_momentum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection (∂_t u, ∂_t v).
        """
        # Vortex-force + Coriolis
        omega_v_m = self.w_flux_y(self.omega[..., 1:-1, :], self.V_m)
        omega_u_m = self.w_flux_x(self.omega[..., 1:-1], self.U_m)

        dt_u = omega_v_m + self.fstar_ugrid[..., 1:-1, :] * self.V_m
        dt_v = -(omega_u_m + self.fstar_vgrid[..., 1:-1] * self.U_m)

        # grad pressure + k_energy
        ke_pressure = self.k_energy + self.p
        dt_u -= torch.diff(ke_pressure, dim=-2) + self.dx_p_ref
        dt_v -= torch.diff(ke_pressure, dim=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self._add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self._add_bottom_drag(dt_u, dt_v)

        return F.pad(dt_u, (0, 0, 1, 1)) * self.masks.u, F.pad(
            dt_v,
            (1, 1, 0, 0),
        ) * self.masks.v

    def advection_h(self) -> torch.Tensor:
        """Advection RHS for thickness perturbation h.

        ∂_t h = - ∇⋅(h_tot u)

        u = [U V]
        h_tot = h_ref + h

        Returns:
            torch.Tensor: h advection (∂_t h).
        """
        h_tot = self.h_ref + self.h
        # Compute (h_tot x V)
        h_tot_flux_y = self.h_flux_y(h_tot, self.V[..., 1:-1])
        # Compute (h_tot x U)
        h_tot_flux_x = self.h_flux_x(h_tot, self.U[..., 1:-1, :])
        # Compute -∇⋅(h_tot u) = ∂_x (h_tot x U) + ∂_y (h_tot x V)
        div_no_flux = -finite_diff.div_nofluxbc(h_tot_flux_x, h_tot_flux_y)
        # Apply h mask
        return div_no_flux * self.masks.h

    def compute_time_derivatives(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dt_u, dt_v, dt_h
        """
        self.compute_diagnostic_variables()
        dt_h = self.advection_h()
        dt_u, dt_v = self.advection_momentum()
        return dt_u, dt_v, dt_h


class SWFilterBarotropic(SW):
    """Shallow Water with Barotropic Filtering."""

    def __init__(self, param: dict[str, Any]) -> None:
        """Instantiate SWFilterBarotropic.

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny),
            unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing,
            x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing,
            y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip,
            inbetween for
                        partial free slip.
            'bottom_drag_coef': float, linear bottom drag coefficient
        """
        super().__init__(param)
        self.tau = 2 * self.dt
        self.barotropic_filter_spectral = param.get(
            "barotropic_filter_spectral",
            False,
        )
        if self.barotropic_filter_spectral:
            verbose.display(
                msg="Using barotropic filter in spectral approximation.",
                trigger_level=2,
            )
            self._set_barotropic_filter_spectral()
        else:
            verbose.display(
                msg="Using barotropic filter in exact form.",
                trigger_level=2,
            )
            self._set_barotropic_filter_exact()

    def _set_barotropic_filter_spectral(self) -> None:
        """Set Helmoltz Solver for barotropic and spectral."""
        self.H_tot = self.H.sum(dim=-3, keepdim=True)
        self.lambd = 1.0 / (self.g * self.dt * self.tau * self.H_tot)
        self.helm_solver = HelmholtzNeumannSolver(
            self.nx,
            self.ny,
            self.dx,
            self.dy,
            self.lambd,
            self.dtype,
            self.device,
            mask=self.masks.h[0, 0],
        )

    def _set_barotropic_filter_exact(self) -> None:
        """Set Helmoltz Solver for barotropic and exact form."""
        coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0, 0]
        coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0, 0]
        lambd = 1.0 / (self.g * self.dt * self.tau)
        self.helm_solver = MG_Helmholtz(
            self.dx,
            self.dy,
            self.nx,
            self.ny,
            coef_ugrid=coef_ugrid,
            coef_vgrid=coef_vgrid,
            lambd=lambd,
            device=self.device,
            dtype=self.dtype,
            mask=self.masks.h[0, 0],
            niter_bottom=20,
            use_compilation=False,
        )

    def filter_barotropic_waves(
        self,
        dt_u: torch.Tensor,
        dt_v: torch.Tensor,
        dt_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inspired from https://doi.org/10.1029/2000JC900089.

        Args:
            dt_u (torch.Tensor): Derivative of prognostic variable u
            dt_v (torch.Tensor): Derivative of prognostic variable v
            dt_h (torch.Tensor): Derivative of prognostic variable h

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: filtered dt_u,
            filtered dt_v, dt_h
        """
        # compute RHS
        u_star = (self.u + self.dt * dt_u) / self.dx
        v_star = (self.v + self.dt * dt_v) / self.dy
        u_bar_star = (u_star * self.h_tot_ugrid).sum(
            dim=-3,
            keepdim=True,
        ) / self.h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * self.h_tot_vgrid).sum(
            dim=-3,
            keepdim=True,
        ) / self.h_tot_vgrid.sum(dim=-3, keepdim=True)
        if self.barotropic_filter_spectral:
            rhs = (
                1.0
                / (self.g * self.dt * self.tau)
                * (
                    torch.diff(u_bar_star, dim=-2) / self.dx
                    + torch.diff(v_bar_star, dim=-1) / self.dy
                )
            )
            w_surf_imp = self.helm_solver.solve(rhs)
        else:
            rhs = (
                1.0
                / (self.g * self.dt * self.tau)
                * (
                    torch.diff(self.h_tot_ugrid * u_bar_star, dim=-2) / self.dx
                    + torch.diff(self.h_tot_vgrid * v_bar_star, dim=-1)
                    / self.dy
                )
            )
            coef_ugrid = (self.h_tot_ugrid * self.masks.u)[0, 0]
            coef_vgrid = (self.h_tot_vgrid * self.masks.v)[0, 0]
            w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)
            # WIP

        filt_u = (
            F.pad(
                -self.g * self.tau * torch.diff(w_surf_imp, dim=-2),
                (0, 0, 1, 1),
            )
            * self.masks.u
        )
        filt_v = (
            F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-1), (1, 1))
            * self.masks.v
        )

        return dt_u + filt_u, dt_v + filt_v, dt_h

    def compute_time_derivatives(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time derivatives of prognostic variables.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  dt_u, dt_v, dt_h
        """
        dt_u, dt_v, dt_h = super().compute_time_derivatives()
        dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h)
        return dt_u, dt_v, dt_h
