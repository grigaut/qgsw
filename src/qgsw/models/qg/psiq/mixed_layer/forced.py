"""Forced QGPSIQSST models."""

from __future__ import annotations

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
from qgsw.models.io import IO
from qgsw.models.qg.psiq.mixed_layer.core import QGPSIQSSTCore
from qgsw.models.qg.stretching_matrix import compute_A_tilde
from qgsw.solver.finite_diff import laplacian
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils.reshaping import crop

if TYPE_CHECKING:
    from qgsw.decomposition.base import SpaceTimeDecomposition
    from qgsw.decomposition.supports.space.base import SpaceSupportFunction
    from qgsw.decomposition.supports.time.base import TimeSupportFunction
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


class QGPSIQSSTRGSI(QGPSIQSSTCore[PSIQSSTTAlpha, StatePSIQSSTAlpha]):
    """QG model with mixed layer and Psi2 transport with deformation radius."""

    _basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

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
        self._fpsi2 = basis.localize(space.psi.xy.x, space.psi.xy.y)

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

    def compute_psi_2_dt(self, time: torch.Tensor) -> torch.Tensor:
        """Compute contribution of ѱ₂'s time derivative.

        Args:
            time (torch.Tensor): Time to evaluate at.

        Returns:
            torch.Tensor: -f₀²ѱ₂/H₂g₂
        """
        dt_psi2 = self._fpsi2.dt(time)
        return (self.beta_plane.f0**2) * self._A12 * dt_psi2

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
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat([self._curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self._curl_tau, self.zeros_inside, bottom_drag],
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
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl - 1 == 2:
            fcg_drag = torch.cat([self._curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self._curl_tau, self.zeros_inside, bottom_drag],
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
        dq = (
            -div_flux_q
            + fcg_drag
            + self.beta_plane.f0 / self.H[:1] * (e[:, :-1] - e[:, 1:])
        ) * self.masks.h

        dt_psi2 = self.compute_psi_2_dt(self._substep_time)
        dq_i = self._interpolate(dq) + crop(dt_psi2, 1)

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

        temp_1_anom = torch.mean(sst_anom * self.masks.h) - self.temp_1_offset

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
            with_2nd_order=True,
            with_4th_order=True,
        )

        dsst = (
            -div_flux_sst
            + self._wek * sst_anom / self.H_ml
            + heat_flux
            + fluxes
            + diffusion
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
        dq = (
            -div_flux_q
            + fcg_drag
            + self.beta_plane.f0 / self.H[:1] * (e[:, :-1] - e[:, 1:])
        ) * self.masks.h
        dt_psi2 = self.compute_psi_2_dt(self._substep_time)
        dq_i = self._interpolate(dq) + crop(dt_psi2, 1)

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

        temp_1_anom = torch.mean(sst_anom * self.masks.h) - self.temp_1_offset

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
            with_2nd_order=True,
            with_4th_order=True,
        )
        dsst = (
            -div_flux_sst
            + self._wek * sst_anom / self.H_ml
            + heat_flux
            + fluxes
            + diffusion
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
        temp_1_anom = torch.mean(sst_anom * self.masks.h) - self.temp_1_offset
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
