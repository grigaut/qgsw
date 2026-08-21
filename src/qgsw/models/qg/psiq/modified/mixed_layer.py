"""QG model with a mixed layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch

from qgsw.exceptions import InvalidLayerNumberError
from qgsw.fields.variables.state import (
    StatePSIQSST,
)
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQSST,
    PSIQSSTT,
    BasePSIQ,
)
from qgsw.logging import getLogger
from qgsw.models.core import time_steppers
from qgsw.models.io import IO
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.variable_sets import QGPSIQVariableSet
from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.solver.boundary_conditions.base import Boundaries
    from qgsw.solver.pv_inversion import (
        BasePVInversion,
    )
    from qgsw.utils.interpolation import LinearInterpolation

T = TypeVar("T", bound=BasePSIQ)


logger = getLogger(__name__)


class QGPSIQMLCore(QGPSIQCore[PSIQSSTT, StatePSIQSST]):
    """Finite volume multi-layer QG solver with mixed layer."""

    H_ml = torch.tensor(100, **defaults.get())  # Mixed layer depth in meters
    K2 = torch.tensor(380, **defaults.get())  # See Kravtsov, 2022
    temp_1_offset = 2
    delta_temp_1 = 8

    @property
    def psi(self) -> torch.Tensor:
        """StatePSIQ Variable psi: Stream function.

        └── (n_ens, nl, nx+1,ny+1)-shaped.
        """
        return self._state.psi.get()

    @property
    def q(self) -> torch.Tensor:
        """StatePSIQ Variable q: Potential Vorticity.

        └── (n_ens, nl, nx, ny)-shaped.
        """
        return self._state.q.get()

    @property
    def sst(self) -> torch.Tensor:
        """StatePSIQ Variable sst: Sea Surface Temperature.

        └── (n_ens, nl, nx, ny)-shaped.
        """
        return self._state.sst.get()

    @property
    def q_anom(self) -> torch.Tensor:
        """Potential Vorticity anomaly.

        └── (n_ens, nl, nx, ny)-shaped.
        """
        return self.q - self._beta_effect

    @property
    def vorticity(self) -> torch.Tensor:
        """Vorticity.

        └── (n_ens, nl, nx, ny)-shaped.
        """
        return self._compute_vort_from_psi(self.psi)

    @property
    def with_bc(self) -> bool:
        """Whether an inhomogeneous solver is used or not."""
        return self._with_bc

    @property
    def solver(self) -> BasePVInversion:
        """Solver for PVInversion."""
        return (
            self._solver_inhomogeneous
            if self.with_bc
            else self._solver_homogeneous
        )

    @property
    def mean_flow(self) -> PSIQ:
        """Mean flow."""
        if not self._with_mean_flow:
            msg = "No mean flow specified."
            raise ValueError(msg)
        return PSIQ(
            psi=self._sf_bar,
            q=self._pv_bar,
        )

    @property
    def perturbation(self) -> PSIQ:
        """Perturbation."""
        if not self._with_mean_flow:
            msg = "No mean flow specified."
            raise ValueError(msg)
        return PSIQ(self.psi, self.q) - self.mean_flow

    def _set_io(self, state: StatePSIQSST) -> None:
        self._io = IO(state.t, state.psi, state.q, state.sst)

    def _set_state(self) -> None:
        """Set the state."""
        with torch.no_grad():
            self._state = StatePSIQSST.steady(
                n_ens=self.n_ens,
                nl=self.space.nl,
                nx=self.space.nx,
                ny=self.space.ny,
                dtype=self.dtype,
                device=self.device.get(),
            )
        self._set_io(self._state)
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

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
        super().set_wind_forcing(taux, tauy)
        factor = 1 / (self.beta_plane.f0 * self.H_ml)
        self._uw = self._tauy.unsqueeze(0).unsqueeze(0) * factor
        self._vw = -self._taux.unsqueeze(0).unsqueeze(0) * factor
        self._wek = self.H[0] * self._curl_tau / self.beta_plane.f0

    def set_boundary_maps(
        self,
        sf_bc_interp: LinearInterpolation[Boundaries],
        pv_bc_interp: LinearInterpolation[Boundaries],
        sst_bc_interp: LinearInterpolation[Boundaries],
    ) -> None:
        """Set the boundary maps.

        Args:
            sf_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for stream function at locations
                (imin,imax+1,jmin,jmax+1).
            pv_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for potential vorticity at locations
                (imin,imax,jmin,jmax).
            sst_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for sea surface temperature at locations
                (imin,imax,jmin,jmax).
        """
        self._switch_to_inhomogeneous()
        self._sf_bc_interp = sf_bc_interp
        self._pv_bc_interp = pv_bc_interp
        self._sst_bc_interp = sst_bc_interp
        self._set_boundaries(self.time.item())

    def _set_boundaries(self, time: float) -> None:
        """Set the boundaries to match given time.

        Args:
            time (float): Time.
        """
        sf_bc = self._sf_bc_interp(time)
        if self._with_mean_flow:
            self._sf_bar = self._sf_bar_interp(time)
            sf_bc -= self._sf_bar_bc_interp(time)

        self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

        pv_bc = self._pv_bc_interp(time)
        sst_bc = self._sst_bc_interp(time)
        if self.wide:
            if pv_bc.width != 3:
                msg = "For wide boundary, pv_bc must be 3 points wide."
                raise ValueError(msg)
            self._pv_bc = pv_bc
            self._sst_bc = sst_bc
            if self._with_mean_flow:
                self._pv_bar = self._pv_bar_interp(time)
                self._pv_bar_bc = self._pv_bar_bc_interp(time)
                self._pv_bc -= self._pv_bar_bc
                self._sst_bar = self._sst_bar_interp(time)
                self._sst_bar_bc = self._sst_bar_bc_interp(time)
                self._sst_bc -= self._sst_bar_bc

        else:
            self._pv_bc = pv_bc.get_band(0)
            if self._with_mean_flow:
                self._pv_bar = self._pv_bar_interp(time)
                self._pv_bar_bc = self._pv_bar_bc_interp(time).get_band(0)
                self._pv_bc -= self._pv_bar_bc

    def set_mean_flow(
        self,
        sf_bar_interp: LinearInterpolation[torch.Tensor],
        pv_bar_interp: LinearInterpolation[torch.Tensor],
        sst_bar_interp: LinearInterpolation[torch.Tensor],
        sf_bar_bc_interp: LinearInterpolation[Boundaries],
        pv_bar_bc_interp: LinearInterpolation[Boundaries],
        sst_bar_bc_interp: LinearInterpolation[Boundaries],
    ) -> None:
        """Set the mean flow.

        Args:
            sf_bar_interp (LinearInterpolation[torch.Tensor]): Mean stream
                function flow.
            pv_bar_interp (LinearInterpolation[torch.Tensor]): Associated mean
                potential vorticity flow.
            sst_bar_interp (LinearInterpolation[torch.Tensor]): Mean
                SST flow.
            sf_bar_bc_interp (LinearInterpolation[Boundaries]): Boundary
                conditions for stream function's mean flow.
            pv_bar_bc_interp (LinearInterpolation[Boundaries]): Boundary
                conditions for potential vorticity's mean flow.
            sst_bar_bc_interp (LinearInterpolation[Boundaries]): Boundary
                conditions for SST's mean flow.
        """
        raise NotImplementedError
        if not self.with_bc:
            msg = (
                "Mean flow only works with inhomogeneous boundary conditions."
            )
            raise ValueError(msg)
        self._with_mean_flow = True
        self._sf_bar_interp = sf_bar_interp
        self._pv_bar_interp = pv_bar_interp
        self._sst_bar_interp = sst_bar_interp
        self._sf_bar_bc_interp = sf_bar_bc_interp
        self._pv_bar_bc_interp = pv_bar_bc_interp
        self._sst_bar_bc_interp = sst_bar_bc_interp
        self._set_boundaries(self.time.item())

    def _compute_advection_homogeneous(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        tracer: torch.Tensor,
    ) -> torch.Tensor:
        return self.div_flux(
            tracer,
            u[..., 1:-1, :],
            v[..., 1:-1],
        )

    def _compute_sst_advection_homogeneous(
        self, psi: torch.Tensor, sst: torch.Tensor
    ) -> torch.Tensor:
        """Compute advection pv advection for homogeneous problem.

        Args:
            psi (torch.Tensor): Top layer stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped
            sst (torch.Tensor): Surface stream function.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: RHS: ∇·(u_ML x SST)
                └──  (n_ens, nl, nx, ny)-shaped
        """
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        return self.div_flux(
            sst,
            (u[:, :1] + self._uw)[..., 1:-1, :],
            (v[:, :1] + self._vw)[..., 1:-1],
        )

    def _compute_advection_inhomogeneous(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        tracer: torch.Tensor,
        tracer_bc: Boundaries,
    ) -> torch.Tensor:
        tracer_with_bc = tracer_bc.expand(tracer)
        return self.div_flux(tracer_with_bc, u, v)

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
        if self.space.nl == 1:
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl == 2:
            fcg_drag = torch.cat([self._curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self._curl_tau, self.zeros_inside, bottom_drag],
                dim=-3,
            )
        return fcg_drag

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
        if self.space.nl == 1:
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl == 2:
            fcg_drag = torch.cat([self._curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self._curl_tau, self.zeros_inside, bottom_drag],
                dim=-3,
            )
        return fcg_drag

    def compute_time_derivatives(self, prognostic: PSIQSST) -> PSIQSST:
        """Compute time derivatives.

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
        if self.with_bc:
            return self._compute_time_derivatives_inhomogeneous(prognostic)
        return self._compute_time_derivatives_homogeneous(prognostic)

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
        psi, q, sst = prognostic
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        div_flux_q = self._compute_advection_homogeneous(u, v, q)
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)

        e = self.compute_entrainments(sst)
        dq = (
            -div_flux_q
            + fcg_drag
            + self.beta_plane.f0 / self.H * (e[:, :-1] - e[:, 1:])
        ) * self.masks.h

        u_ml = u[:, :1] + self._uw * self.masks.u
        v_ml = v[:, :1] + self._vw * self.masks.v

        div_flux_sst = self._compute_advection_homogeneous(u_ml, v_ml, sst)

        temp_1 = torch.mean(sst * self.masks.h) - self.temp_1_offset

        heat_flux = torch.where(
            self._wek > 0,
            -self._wek * (sst - temp_1) / self.H_ml,
            0,
        )
        diffusion = (
            laplacian(
                torch.nn.functional.pad(
                    sst,
                    (1, 1, 1, 1),
                    mode="replicate",
                ),
                self.space.dx,
                self.space.dy,
            )
            * self.K2
        )
        dsst = (
            -div_flux_sst + self._wek * sst / self.H_ml + heat_flux + diffusion
        ) * self.masks.h
        dq_i = self._interpolate(dq)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        return PSIQSST(dpsi, dq, dsst)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

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
        psi_i, q_i, sst = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        div_flux_q = self._compute_advection_inhomogeneous(
            u, v, q, self._pv_bc
        )
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-div_flux_q + fcg_drag) * self.masks.h

        u_ml = u[:, :1] + self._uw * self.masks.u
        v_ml = v[:, :1] + self._vw * self.masks.v

        div_flux_sst = self._compute_advection_inhomogeneous(
            u_ml, v_ml, sst, self._sst_bc
        )

        temp_1 = torch.mean(sst * self.masks.h) - self.temp_1_offset

        heat_flux = torch.where(
            self._wek > 0,
            -self._wek * (sst - temp_1) / self.H_ml,
            0,
        )
        diffusion = (
            laplacian(self._sst_bc.expand(sst), self.space.dx, self.space.dy)
            * self.K2
        )
        dsst = (
            -div_flux_sst + self._wek * sst / self.H_ml + heat_flux + diffusion
        ) * self.masks.h

        dq_i = self._interpolate(dq)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
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

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        return self.set_psi(p[:, :nl] / self.beta_plane.f0)

    def set_q(self, q: torch.Tensor) -> None:
        """Set the value of potential vorticity.

        WARNING: with inhomogeneous boundary condition this might introduce
        errors in ѱ due to interpolation of q.
        You should use the `set_psiq` method instead.

        Args:
            q (torch.Tensor): Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        self.set_q_anomaly(q_anom=q - self._beta_effect)

    def set_q_anomaly(self, q_anom: torch.Tensor) -> None:
        """Set the value of potential vorticity.

        WARNING: with inhomogeneous boundary condition this might introduce
        errors in ѱ due to interpolation of q_anom.
        You should use the `set_psiq` method instead.

        Args:
            q_anom (torch.Tensor): Potential vorticity anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        psi = self.solver.compute_stream_function(self._interpolate(q_anom))
        self._state.update_psiq(PSIQ(psi, q_anom + self._beta_effect))

    def set_psi(self, psi: torch.Tensor) -> None:
        """Set the value of stream function.

        WARNING: with inhomogeneous boundary condition this might introduce
        errors in q due to different beta effects.
        You should use the `set_psiq` method instead.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        q = self._compute_q_from_psi(psi)
        self._state.update_psiq(PSIQ(psi, q))

    def set_sst(self, sst: torch.Tensor) -> None:
        """Set the SST.

        Args:
            sst (torch.Tensor): Sea surface temperature tensor.
        """
        self._state.update_sst(sst)

    def set_psiq(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set both psi and q.

        Args:
            psi (torch.Tensor): Stream function tensor.
            q (torch.Tensor): Potential vorticity tensor.
        """
        self._state.update_psiq(PSIQ(psi, q))

    def set_psiqsst(
        self, psi: torch.Tensor, q: torch.Tensor, sst: torch.Tensor
    ) -> None:
        """Set both psi and q.

        Args:
            psi (torch.Tensor): Stream function tensor.
            q (torch.Tensor): Potential vorticity tensor.
            sst (torch.Tensor): Sea surface temperature tensor.
        """
        self._state.update_psiqsst(PSIQSST(psi, q, sst))

    def set_psiqanom(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set both psi and q.

        Args:
            psi (torch.Tensor): Stream function tensor.
            q (torch.Tensor): Potential vorticity tensor.
        """
        self._state.update_psiq(
            PSIQ(psi, q + self._masks.h * self._beta_effect)
        )

    def compute_entrainments(
        self,
        sst: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entrainments.

        See "Formulation and users’ guide for Q-GCM, Hogg et al, 2014".
        Zero entrainment is assumed for layers below layer 1.

        Args:
            sst (torch.Tensor): Sea surface temperature.

        Returns:
            torch.Tensor: Entrainments vector.
        """
        temp_1 = torch.mean(sst) - self.temp_1_offset
        delta_temp_ml = sst - temp_1
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
            [e_ml, e1 - torch.mean(e1)]
            + [torch.zeros_like(e1) for _ in range(self.space.nl - 1)],
            dim=1,
        )

    def update(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic.

        Args:
            prognostic (PSIQ): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQ: Updated prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            if self._with_mean_flow:
                return self._update_mean_flow(prognostic)
            return self._update_inhomogeneous(prognostic)
        return self._update_homogeneous(prognostic)

    def _timestep(self, prognostic: PSIQSST) -> PSIQSST:
        if self.time_stepper == "rk3":
            self._rk3_step = 0
            psiqsst = time_steppers.rk3_ssp(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        elif self.time_stepper == "euler":
            psiqsst = time_steppers.euler(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        else:
            msg = f"Invalid time stepper: {self.time_stepper}"
            raise ValueError(msg)
        self._state.increment_time(self.dt)
        return psiqsst

    def _update_homogeneous(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQ): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQ: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
        """
        prognostic_i = prognostic
        return self._timestep(prognostic_i)

    def _update_inhomogeneous(self, prognostic: PSIQSST) -> PSIQSST:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQ): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQ: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
        """
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        psi, q, sst = prognostic
        prognostic_i = PSIQ(psi - psi_bc, q, sst)
        psiqsst_i = self._timestep(prognostic_i)
        self._set_boundaries(self.time.item())
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        return PSIQSST(psiqsst_i.psi + psi_bc, psiqsst_i.q, psiqsst_i.sst)

    def _update_mean_flow(self, prognostic: PSIQ) -> PSIQ:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQ): Prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            PSIQ: Updated prognostic variable to advect.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped
        """
        prognostic_pert = prognostic - self.mean_flow
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        prognostic_i = PSIQ(prognostic_pert.psi - psi_bc, prognostic_pert.q)
        psiq_i = self._timestep(prognostic_i)
        self._set_boundaries(self.time.item())
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        psi_bar, q_bar = self.mean_flow
        return PSIQ(psiq_i.psi + psi_bc + psi_bar, psiq_i.q + q_bar)

    @torch.enable_grad()
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.update_psiqsst(self.update(self._state.prognostic.psiqsst))
        t1 = torch.mean(self.sst) - 2
        sst = torch.where(self.sst >= t1, self.sst, t1)
        self._state.update_sst(sst)

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,
        physics: PhysicsConfig,
        model: ModelConfig,
    ) -> dict[str, DiagnosticVariable]:
        """Create variable set.

        Args:
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
            model (ModelConfig): Model configuaration.

        Returns:
            dict[str, DiagnosticVariable]: Variables dictionnary.
        """
        return QGPSIQVariableSet.get_variable_set(space, physics, model)
