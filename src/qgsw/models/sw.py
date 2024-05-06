# ruff: noqa
"""
Shallow-water implementation.
Louis Thiry, Nov 2023 for IFREMER.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from qgsw import verbose
from qgsw.models.core.helmholtz import HelmholtzNeumannSolver
from qgsw.models.core.helmholtz_multigrid import MG_Helmholtz
from qgsw.models.base import Model

from qgsw.models.core import finite_diff, flux


def pool_2d(padded_f: torch.Tensor) -> torch.Tensor:
    """2D pool a padded tensor.

    Args:
        padded_f (torch.Tensor): Tensor to pool.

    Returns:
        torch.Tensor: Padded tensor.
    """
    # average pool padded value
    f_sum_pooled = F.avg_pool2d(
        padded_f,
        (3, 1),
        stride=(1, 1),
        padding=(1, 0),
        divisor_override=1,
    )
    return F.avg_pool2d(
        f_sum_pooled,
        (1, 3),
        stride=(1, 1),
        padding=(0, 1),
        divisor_override=1,
    )


def replicate_pad(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replicate a given pad.

    Args:
        f (torch.Tensor): Tensor to pad.
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: Result
    """
    f_ = F.pad(f, (1, 1, 1, 1))
    mask_ = F.pad(mask, (1, 1, 1, 1))
    mask_sum = pool_2d(mask_)
    f_sum = pool_2d(f_)
    f_out = f_sum / torch.maximum(torch.ones_like(mask_sum), mask_sum)
    return mask_ * f_ + (1 - mask_) * f_out


def reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pytorch cumsum in the reverse order
    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])
    """
    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


def inv_reverse_cumsum(x, dim):
    """Inverse of reverse cumsum function"""
    return torch.cat([-torch.diff(x, dim=dim), x.narrow(dim, -1, 1)], dim=dim)


class SW(Model):
    """# Implementation of multilayer rotating shallow-water model

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

    def __init__(self, param: dict[str, Any]):
        """Parameters

        param: python dict. with following keys
            'nx':       int, number of grid points in dimension x
            'ny':       int, number grid points in dimension y
            'nl':       nl, number of stacked layer
            'dx':       float or Tensor (nx, ny), dx metric term
            'dy':       float or Tensor (nx, ny), dy metric term
            'H':        Tensor (nl,) or (nl, nx, ny), unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing, x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing, y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip, inbetween for
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

    def compute_time_derivatives(self):
        dt_u, dt_v, dt_h = super().compute_time_derivatives()
        return dt_u, dt_v, dt_h

    def step(self):
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

    def compute_diagnostic_variables(self) -> None:
        """Compute the model's diagnostic variables.

        Compute the result given the prognostic
        variables self.u, self.v, self.h .
        """
        self.omega = self.compute_omega(self.u, self.v)
        self.eta = reverse_cumsum(self.h / self.area, dim=-3)
        self.p = torch.cumsum(self.g_prime * self.eta, dim=-3)
        self.U = self.u / self.dx**2
        self.V = self.v / self.dy**2
        self.U_m = self.interp_TP(self.U)
        self.V_m = self.interp_TP(self.V)
        self.k_energy = (
            self.comp_ke(self.u, self.U, self.v, self.V) * self.masks.h
        )

        h_ = replicate_pad(self.h, self.masks.h)
        self.h_ugrid = 0.5 * (h_[..., 1:, 1:-1] + h_[..., :-1, 1:-1])
        self.h_vgrid = 0.5 * (h_[..., 1:-1, 1:] + h_[..., 1:-1, :-1])
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid

    def advection_momentum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection.
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

        dt_h = - div(h_tot [u v])

        h_tot = h_ref + h

        Returns:
            torch.Tensor: h advection.
        """
        h_tot = self.h_ref + self.h
        h_tot_flux_y = self.h_flux_y(h_tot, self.V[..., 1:-1])
        h_tot_flux_x = self.h_flux_x(h_tot, self.U[..., 1:-1, :])
        div_no_flux = -finite_diff.div_nofluxbc(h_tot_flux_x, h_tot_flux_y)
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
    def __init__(self, param: dict[str, Any]):
        super().__init__(param)
        self.tau = 2 * self.dt
        self.barotropic_filter_spectral = param.get(
            "barotropic_filter_spectral", False
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

    def filter_barotropic_waves(self, dt_u, dt_v, dt_h):
        """Inspired from https://doi.org/10.1029/2000JC900089."""
        # compute RHS
        u_star = (self.u + self.dt * dt_u) / self.dx
        v_star = (self.v + self.dt * dt_v) / self.dy
        u_bar_star = (u_star * self.h_tot_ugrid).sum(
            dim=-3, keepdim=True
        ) / self.h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * self.h_tot_vgrid).sum(
            dim=-3, keepdim=True
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

    def compute_time_derivatives(self):
        dt_u, dt_v, dt_h = super().compute_time_derivatives()
        dt_u, dt_v, dt_h = self.filter_barotropic_waves(dt_u, dt_v, dt_h)
        return dt_u, dt_v, dt_h
