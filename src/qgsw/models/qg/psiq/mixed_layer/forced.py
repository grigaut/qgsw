"""Forced QGPSIQSST models."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.state import (
    StatePSIQSSTAlpha,
)
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQSST,
    PSIQSSTT,
    PSIQSSTTAlpha,
)
from qgsw.logging.core import getLogger
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.mixed_layer.core import QGPSIQSST, QGPSIQSSTCore
from qgsw.models.qg.stretching_matrix import compute_A_tilde
from qgsw.solver.finite_diff import grad, laplacian
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qgsw.spatial.core.grid_conversion import interpolate, interpolate1D
from qgsw.specs import DEVICE, defaults

if TYPE_CHECKING:
    from qgsw.decomposition.base import SpaceTimeDecomposition
    from qgsw.decomposition.supports.space.base import SpaceSupportFunction
    from qgsw.decomposition.supports.time.base import TimeSupportFunction
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D

logger = getLogger(__name__)


class QGPSIQSSTRGSI(QGPSIQSSTCore[PSIQSSTTAlpha, StatePSIQSSTAlpha]):
    """QG model with mixed layer and Psi2 transport with deformation radius."""

    _basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]
    _type = ModelName.QUASI_GEOSTROPHIC_ML
    _sst_forcing: torch.Tensor
    _use_div = False

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,
        beta_plane: BetaPlane,
        g_prime: torch.Tensor,
        optimize: bool = True,
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor.
                └── (nl,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_2d=space_2d,
            H=H,
            beta_plane=beta_plane,
            g_prime=g_prime,
            optimize=optimize,
        )
        self._A11 = self.A[0, 0]
        self._A12 = self.A[0, 1]
        self.zeros_inside = (
            torch.zeros(
                (self.n_ens, self.space.nl - 3, self.space.nx, self.space.ny),
                defaults.get(),
            )
            if (self.space.nl - 3) > 0
            else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        try:
            return self._state.alpha.get()
        except AttributeError:
            return torch.tensor(0, **defaults.get())

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)
        self.compute_auxillary_matrices()
        self._set_solver()

    @property
    def wide_space(self) -> SpaceDiscretization2D:
        """Wide space."""
        return self._wide_space

    @wide_space.setter
    def wide_space(self, space: SpaceDiscretization2D) -> None:
        self._wide_space = space

    @property
    def basis(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Decomposition basis."""
        return self._basis

    @basis.setter
    def basis(
        self,
        basis: SpaceTimeDecomposition[
            SpaceSupportFunction, TimeSupportFunction
        ],
    ) -> None:
        self._basis = basis
        space = self.space.remove_h()
        self._fpsi2 = basis.localize(space.q.xy.x, space.q.xy.y)
        with contextlib.suppress(AttributeError):
            self._fpsi2_wide = basis.localize(
                self.wide_space.q.xy.x,
                self.wide_space.q.xy.y,
            )
        self._fpsi2_dx = basis.localize_dx(space.u.xy.x, space.u.xy.y)
        self._fpsi2_dy = basis.localize_dy(space.v.xy.x, space.v.xy.y)

    @property
    def sst_forcing(self) -> torch.Tensor:
        """SST forcing term.

        └── (n_ens, nl, nx, ny)-shaped
        """
        try:
            return self._sst_forcing
        except AttributeError:
            return torch.zeros_like(self.q)

    @sst_forcing.setter
    def sst_forcing(self, sst_forcing: torch.Tensor) -> None:
        self._sst_forcing = sst_forcing

    def _set_io(self, state: StatePSIQSSTAlpha) -> None:
        self._io = IO(state.t, state.psi, state.q, state.sst, state.alpha)

    def _set_state(self) -> None:
        """Set the state."""
        alpha = torch.tensor(0, **defaults.get())
        self._state = StatePSIQSSTAlpha.from_tensors(
            *PSIQSSTT.steady(
                n_ens=self.n_ens,
                nl=self.space.nl - 1,
                nx=self.space.nx,
                ny=self.space.ny,
                dtype=self.dtype,
                device=self.device.get(),
            ),
            alpha,
        )
        self._sst_mean = (self._state.sst.get() * self.masks.h).mean()
        self._state.update_sst(self._state.sst.get() - self._sst_mean)
        self.compute_auxillary_matrices()
        self._set_solver()
        self._set_io(self._state)
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self._solver_homogeneous = HomogeneousPVInversion(
            self.A[:1, :1],
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversion(
            self.A[:1, :1],
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        if self._with_bc:
            sf_bc = self._sf_bc_interp(self.time.item())
            if self._with_mean_flow:
                sf_bar_bc = self._sf_bar_bc_interp(self.time.item())
                self._solver_inhomogeneous.set_boundaries(
                    sf_bc.get_band(0) - sf_bar_bc.get_band(0)
                )
            else:
                self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def compute_auxillary_matrices(self) -> None:
        """Compute auxillary matrices."""
        H = self.H[:, 0, 0]
        g_prime = self.g_prime[:, 0, 0]

        self.A = compute_A_tilde(H, g_prime, self.alpha, **defaults.get())
        self._A11 = self.A[:1, :1]
        self._A12 = self.A[:1, 1:2]

    def compute_forcing_analytical(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing using analytical spatial derivatives of psi2.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²/H₂g₂[∂ₜѱ₂ + J(ѱ₁, ѱ₂)]
        """
        u, v = self._grad_perp(psi1)
        u /= self.space.dy
        v /= self.space.dx

        dt_psi2 = self._fpsi2.dt(time)
        dx_psi2 = self._fpsi2_dx(time)
        dy_psi2 = self._fpsi2_dy(time)

        u_dxpsi2 = u * dx_psi2
        v_dypsi2 = v * dy_psi2

        adv = (u_dxpsi2[..., 1:, :] + u_dxpsi2[..., :-1, :]) / 2 + (
            v_dypsi2[..., 1:] + v_dypsi2[..., :-1]
        ) / 2
        return (self.beta_plane.f0**2) * self._A12 * (dt_psi2 + adv)

    def compute_forcing(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²/H₂g₂[∂ₜѱ₂ + J(ѱ₁, ѱ₂)]
        """
        if self._use_div and self.with_bc:
            return self.compute_forcing_div(time, psi1)
        return self.compute_forcing_analytical(time, psi1)

    def compute_forcing_div(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing using divergence of wide psi2.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²/H₂g₂[∂ₜѱ₂ + J(ѱ₁, ѱ₂)]
        """
        dt_psi2 = self._fpsi2.dt(time)

        u, v = self._grad_perp(psi1)
        u /= self.space.dy
        v /= self.space.dx
        psi2 = self._fpsi2_wide(time)
        adv = self.div_flux(psi2, u, v)

        return (self.beta_plane.f0**2) * self._A12 * (dt_psi2 + adv)

    def _switch_to_inhomogeneous(self) -> None:
        if (not self.with_bc) and self._use_div:
            msg = "Will use flux divergence for forcing advection."
            logger.detail(msg)
        return super()._switch_to_inhomogeneous()

    def _compute_q_anom_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        vort = self._compute_vort_from_psi(psi)
        stretching = self.beta_plane.f0**2 * self._A11 * psi
        if self.with_bc:
            return vort - self.masks.h * self._interpolate(stretching)
        return vort - self.masks.h * self._interpolate(
            self.masks.psi * stretching
        )

    def _compute_drag_inhomogeneous(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute wind and bottom drag contribution.

        Args:
            psi (torch.Tensor): Stream function.
                └──  psi: (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Wind and bottom drag.
                └──  (n_ens, nl, nx, ny)-shaped
        """
        sf_boundary = self._sf_bc_interp(self.time.item())
        sf_wide = sf_boundary.expand(psi[..., 1:-1, 1:-1])
        omega = interpolate(laplacian(sf_wide, self.space.dx, self.space.dy))
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl - 1 == 1:
            fcg_drag = bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat(
                [torch.zeros_like(bottom_drag), bottom_drag], dim=-3
            )
        else:
            fcg_drag = torch.cat(
                [
                    torch.zeros_like(bottom_drag),
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )
        return fcg_drag

    def _compute_drag_homogeneous(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute wind and bottom drag contribution.

        Args:
            psi (torch.Tensor): Stream function.
                └──  psi: (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Wind and bottom drag.
                └──  (n_ens, nl, nx, ny)-shaped
        """
        omega = self._interpolate(
            self._laplacian_h(psi, self.space.dx, self.space.dy)
            * self.masks.psi,
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl - 1 == 1:
            fcg_drag = bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat(
                [torch.zeros_like(bottom_drag), bottom_drag], dim=-3
            )
        else:
            fcg_drag = torch.cat(
                [
                    torch.zeros_like(bottom_drag),
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )
        return fcg_drag

    def _compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
                └──  sst : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq, sst
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
                └──  dsst : (n_ens, nl, nx, ny)-shaped
        """
        psi, q, sst_anom = prognostic
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        e = self.compute_entrainments(sst_anom)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (
            -div_flux_q
            + fcg_drag
            + forcing
            + self.beta_plane.f0 / self.H[:1] * (e[:, :-1] - e[:, 1:])
        ) * self.masks.h
        dq_i = self._interpolate(dq)
        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )

        ## Compute dSST
        u_ml = u[:, :1] + self._uw * self.masks.u
        v_ml = v[:, :1] + self._vw * self.masks.v

        div_flux_sst = self._compute_advection_homogeneous(
            u_ml,
            v_ml,
            sst_anom,
        )

        temp_1_anom = self._compute_temp1_anom(self.sst_anom)

        heat_flux = torch.where(
            self._wek > 0,
            -self._wek * (sst_anom - temp_1_anom) / self.H_ml,
            0,
        )

        fluxes = self.compute_fluxes(
            sst_anom,
            with_atm_convective=False,
            with_radiative=False,
        )
        diffusion = self.compute_diffusion(
            sst_anom,
            with_2nd_order=False,
            with_4th_order=False,
        )

        forcing = self.compute_sst_forcing_inhomogeneous(sst_anom)

        dsst = (
            -div_flux_sst
            + self._wek * sst_anom / self.H_ml
            + heat_flux
            + fluxes
            + diffusion
            + forcing
        ) * self.masks.h
        return PSIQSST(dpsi, dq, dsst)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQSST): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i, sst_anom = prognostic

        ## Reconstruct ψ and q
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_inhomogeneous(
            u, v, q, self._pv_bc
        )
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        e = self.compute_entrainments(sst_anom)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (
            -div_flux_q
            + fcg_drag
            + forcing
            + self.beta_plane.f0 / self.H[:1] * (e[:, :-1] - e[:, 1:])
        ) * self.masks.h
        dq_i = self._interpolate(dq)

        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )

        ## Compute dSST
        u_ml = u[:, :1] + self._uw * self.masks.u
        v_ml = v[:, :1] + self._vw * self.masks.v

        div_flux_sst = self._compute_advection_inhomogeneous(
            u_ml,
            v_ml,
            sst_anom,
            self._sst_bc,
        )

        temp_1_anom = self._compute_temp1_anom(self.sst_anom)

        heat_flux = torch.where(
            self._wek > 0,
            -self._wek * (sst_anom - temp_1_anom) / self.H_ml,
            0,
        )
        fluxes = self.compute_fluxes(
            sst_anom,
            with_atm_convective=False,
            with_radiative=False,
        )
        diffusion = self.compute_diffusion(
            sst_anom,
            self._sst_bc,
            with_2nd_order=False,
            with_4th_order=False,
        )

        forcing = self.compute_sst_forcing_inhomogeneous(sst_anom)

        dsst = (
            -div_flux_sst
            + self._wek * sst_anom / self.H_ml
            + heat_flux
            + fluxes
            + diffusion
            + forcing
        ) * self.masks.h

        ## Adjust boundaries
        if self.time_stepper == "rk3":
            # Boundary condition interpolation
            self._rk3_step += 1
            if self._rk3_step == 1:
                coef = 1
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 2:
                coef = 1 / 2
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 3:
                # There won't be any additional step.
                ...
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
        return PSIQSST(dpsi, dq, dsst)

    def compute_entrainments(
        self,
        sst_anom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entrainments.

        See "Formulation and users’ guide for Q-GCM, Hogg et al, 2014".
        Zero entrainment is assumed for layers below layer 1.

        Args:
            sst_anom (torch.Tensor): Sea surface temperature.

        Returns:
            torch.Tensor: Entrainments vector.
        """
        temp_1_anom = self._compute_temp1_anom(self.sst_anom)
        delta_temp_ml = sst_anom - temp_1_anom
        e_ml = self._wek
        e1 = torch.where(
            self._wek > 0, 0, delta_temp_ml / self.delta_temp_1 * self._wek
        )
        e1 += torch.where(
            delta_temp_ml >= 0,
            0,
            self.H_ml / self.dt * delta_temp_ml / self.delta_temp_1,
        )
        return torch.cat(
            [e_ml, e1 - torch.mean(e1)],
            dim=1,
        )

    def compute_sst_forcing_inhomogeneous(
        self, sst: torch.Tensor
    ) -> torch.Tensor:
        """SST forcing."""
        sst_ = self._sst_bc.get_band(0).expand(sst)
        dx_sst, dy_sst = grad(sst_)
        dx_sst /= self.space.dx
        dy_sst /= self.space.dy

        dx_sst = interpolate1D(dx_sst, dim=-1)
        dy_sst = interpolate1D(dy_sst, dim=-2)

        grad_sst_norm = (dx_sst.square() + dy_sst.square()).sqrt()

        return -self.sst_forcing * interpolate(grad_sst_norm)

    def compute_sst_forcing_homogeneous(
        self, sst: torch.Tensor
    ) -> torch.Tensor:
        """SST forcing."""
        sst_ = torch.nn.functional.pad(sst, (1, 1, 1, 1), value=0)
        dx_sst, dy_sst = grad(sst_)
        dx_sst /= self.space.dx
        dy_sst /= self.space.dy

        dx_sst = interpolate1D(dx_sst, dim=-1)
        dy_sst = interpolate1D(dy_sst, dim=-2)

        grad_sst_norm = (dx_sst.square() + dy_sst.square()).sqrt()

        return -self.sst_forcing * interpolate(grad_sst_norm)


class QGPSIQSSTAdvRGSI(QGPSIQSSTRGSI):
    """QGPSIQSSTRGSI with advected SST."""

    _H_ml = None
    _temp_1_offset = None

    def set_wind_forcing(
        self,
        taux: torch.Tensor | float,
        tauy: torch.Tensor | float,
    ) -> None:
        """Set the wind forcing.

        Args:
            taux (torch.Tensor): Wind stress in the x direction.
                └── (n_ens, nl, nx, ny)-shaped
            tauy (torch.Tensor): Wind stress in the y direction.
                └── (n_ens, nl, nx, ny)-shaped
        """
        QGPSIQCore.set_wind_forcing(self, taux, tauy)

    def _compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
                └──  sst : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq, sst
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
                └──  dsst : (n_ens, nl, nx, ny)-shaped
        """
        psi, q, sst_anom = prognostic
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (-div_flux_q + fcg_drag + forcing) * self.masks.h
        dq_i = self._interpolate(dq)
        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )

        ## Compute dSST
        u_ml = u[:, :1]
        v_ml = v[:, :1]

        div_flux_sst = self._compute_advection_homogeneous(
            u_ml,
            v_ml,
            sst_anom,
        )

        diffusion = self.compute_diffusion(
            sst_anom,
            with_2nd_order=False,
            with_4th_order=False,
        )

        forcing = self.compute_sst_forcing_inhomogeneous(sst_anom)

        dsst = (-div_flux_sst + diffusion + forcing) * self.masks.h
        return PSIQSST(dpsi, dq, dsst)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQSST): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i, sst_anom = prognostic

        ## Reconstruct ψ and q
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_inhomogeneous(
            u, v, q, self._pv_bc
        )
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (-div_flux_q + fcg_drag + forcing) * self.masks.h
        dq_i = self._interpolate(dq)

        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )

        ## Compute dSST
        u_ml = u[:, :1]
        v_ml = v[:, :1]

        div_flux_sst = self._compute_advection_inhomogeneous(
            u_ml,
            v_ml,
            sst_anom,
            self._sst_bc,
        )

        diffusion = self.compute_diffusion(
            sst_anom,
            self._sst_bc,
            with_2nd_order=False,
            with_4th_order=False,
        )
        forcing = self.compute_sst_forcing_inhomogeneous(sst_anom)
        dsst = (-div_flux_sst + diffusion + forcing) * self.masks.h

        ## Adjust boundaries
        if self.time_stepper == "rk3":
            # Boundary condition interpolation
            self._rk3_step += 1
            if self._rk3_step == 1:
                coef = 1
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 2:
                coef = 1 / 2
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 3:
                # There won't be any additional step.
                ...
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
        return PSIQSST(dpsi, dq, dsst)

    @torch.enable_grad()
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.update_psiqsst(self.update(self._state.prognostic.psiqsst))

    def compute_entrainments(
        self,
        sst_anom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entrainments.

        See "Formulation and users’ guide for Q-GCM, Hogg et al, 2014".
        Zero entrainment is assumed for layers below layer 1.

        Args:
            sst_anom (torch.Tensor): Sea surface temperature.

        Returns:
            torch.Tensor: Entrainments vector.
        """
        msg = "This method is of no use with this model."
        raise NotImplementedError(msg)

    def compute_sst_forcing_inhomogeneous(
        self, sst: torch.Tensor
    ) -> torch.Tensor:
        """SST forcing."""
        sst_ = self._sst_bc.get_band(0).expand(sst)
        dx_sst, dy_sst = grad(sst_)
        dx_sst /= self.space.dx
        dy_sst /= self.space.dy

        dx_sst = interpolate1D(dx_sst, dim=-1)
        dy_sst = interpolate1D(dy_sst, dim=-2)

        grad_sst_norm = (dx_sst.square() + dy_sst.square()).sqrt()

        return -self.sst_forcing * interpolate(grad_sst_norm)

    def compute_sst_forcing_homogeneous(
        self, sst: torch.Tensor
    ) -> torch.Tensor:
        """SST forcing."""
        sst_ = torch.nn.functional.pad(sst, (1, 1, 1, 1), value=0)
        dx_sst, dy_sst = grad(sst_)
        dx_sst /= self.space.dx
        dy_sst /= self.space.dy

        dx_sst = interpolate1D(dx_sst, dim=-1)
        dy_sst = interpolate1D(dy_sst, dim=-2)

        grad_sst_norm = (dx_sst.square() + dy_sst.square()).sqrt()

        return -self.sst_forcing * interpolate(grad_sst_norm)


class QGPSIQSSTForced(QGPSIQSST):
    """Forced RG model with SST Advection."""

    _forcing: torch.Tensor

    _H_ml = None
    _temp_1_offset = None

    @property
    def forcing(self) -> torch.Tensor:
        """Forcing term.

        └── (n_ens, nl, nx, ny)-shaped
        """
        try:
            return self._forcing
        except AttributeError:
            return torch.zeros_like(self.q)

    @forcing.setter
    def forcing(self, forcing: torch.Tensor) -> None:
        self._forcing = forcing

    @property
    def sst_forcing(self) -> torch.Tensor:
        """SST forcing term.

        └── (n_ens, nl, nx, ny)-shaped
        """
        try:
            return self._sst_forcing
        except AttributeError:
            return torch.zeros_like(self.q)

    @sst_forcing.setter
    def sst_forcing(self, sst_forcing: torch.Tensor) -> None:
        self._sst_forcing = sst_forcing

    @property
    def wind_scaling(self) -> torch.Tensor:
        """Wind forcing scaling."""
        try:
            return self._wind_scaling
        except AttributeError:
            return self.H[0, 0, 0].item()

    @wind_scaling.setter
    def wind_scaling(self, wind_scaling: torch.Tensor) -> None:
        self._wind_scaling = wind_scaling

    def set_wind_forcing(
        self,
        taux: torch.Tensor | float,
        tauy: torch.Tensor | float,
    ) -> None:
        """Set the wind forcing.

        WARNING: Both taux and tauy are padded on the right.

        Args:
            taux (torch.Tensor): Wind stress in the x direction.
                └── (n_ens, nl, nx, ny)-shaped
            tauy (torch.Tensor): Wind stress in the y direction.
                └── (n_ens, nl, nx, ny)-shaped
        """
        if isinstance(taux, float) and isinstance(tauy, float):
            self._curl_tau = torch.zeros(
                (self.n_ens, 1, self.space.nx, self.space.ny),
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            return
        curl_tau = (
            torch.diff(tauy, dim=-2) / self._space.dx
            - torch.diff(taux, dim=-1) / self._space.dy
        )
        self._curl_tau = curl_tau.unsqueeze(0).unsqueeze(0) / self.wind_scaling

    def _compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQSST): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
                └──  sst : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq, sst
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
                └──  dsst : (n_ens, nl, nx, ny)-shaped
        """
        psi, q, sst_anom = prognostic
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        dq = (-div_flux_q + fcg_drag + self.forcing) * self.masks.h

        dq_i = self._interpolate(dq)

        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )

        ## Compute dSST
        u_ml = u[:, :1]
        v_ml = v[:, :1]

        div_flux_sst = self._compute_advection_homogeneous(
            u_ml,
            v_ml,
            sst_anom,
        )

        diffusion = self.compute_diffusion(
            sst_anom,
            with_2nd_order=False,
            with_4th_order=False,
        )

        dsst = (-div_flux_sst + diffusion + self.sst_forcing) * self.masks.h
        return PSIQSST(dpsi, dq, dsst)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQSST,
    ) -> PSIQSST:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQSST): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQSST: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i, sst_anom = prognostic

        ## Reconstruct ψ and q
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx

        ## Compute dq
        div_flux_q = self._compute_advection_inhomogeneous(
            u, v, q, self._pv_bc
        )
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-div_flux_q + fcg_drag + self.forcing) * self.masks.h
        dq_i = self._interpolate(dq)

        ## Compute dψ
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )

        ## Compute dSST
        u_ml = u[:, :1]
        v_ml = v[:, :1]

        div_flux_sst = self._compute_advection_inhomogeneous(
            u_ml,
            v_ml,
            sst_anom,
            self._sst_bc,
        )

        diffusion = self.compute_diffusion(
            sst_anom,
            self._sst_bc,
            with_2nd_order=False,
            with_4th_order=False,
        )
        dsst = (-div_flux_sst + diffusion + self.sst_forcing) * self.masks.h

        ## Adjust boundaries
        if self.time_stepper == "rk3":
            # Boundary condition interpolation
            self._rk3_step += 1
            if self._rk3_step == 1:
                coef = 1
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 2:
                coef = 1 / 2
                self._set_boundaries(self.time.item() + coef * self.dt)
            elif self._rk3_step == 3:
                # There won't be any additional step.
                ...
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
        return PSIQSST(dpsi, dq, dsst)

    @torch.enable_grad()
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.update_psiqsst(self.update(self._state.prognostic.psiqsst))

    def compute_entrainments(
        self,
        sst_anom: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entrainments.

        See "Formulation and users’ guide for Q-GCM, Hogg et al, 2014".
        Zero entrainment is assumed for layers below layer 1.

        Args:
            sst_anom (torch.Tensor): Sea surface temperature.

        Returns:
            torch.Tensor: Entrainments vector.
        """
        msg = "This method is of no use with this model."
        raise NotImplementedError(msg)
