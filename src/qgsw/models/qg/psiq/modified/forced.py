"""Forced QG models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from qgsw.fields.variables.state import StatePSIQ, StatePSIQAlpha
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQT,
    PSIQTAlpha,
)
from qgsw.logging.core import getLogger
from qgsw.models.io import IO
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.solver.finite_diff import laplacian
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversionCollinear,
    InhomogeneousPVInversionCollinear,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import DEVICE, defaults
from qgsw.utils.reshaping import crop

if TYPE_CHECKING:
    from qgsw.decomposition.base import SpaceTimeDecomposition
    from qgsw.decomposition.supports.space.base import SpaceSupportFunction
    from qgsw.decomposition.supports.time.base import TimeSupportFunction
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D

logger = getLogger(__name__)


class QGPSIQForced(QGPSIQCore[PSIQT, StatePSIQ]):
    """Specify and additional forcing term."""

    _forcing: torch.Tensor = None

    @property
    def forcing(self) -> torch.Tensor:
        """Forcing term.

        └── (n_ens, nl, nx, ny)-shaped
        """
        if self._forcing is None:
            return torch.zeros_like(self.q)
        return self._forcing

    @forcing.setter
    def forcing(self, forcing: torch.Tensor) -> None:
        self._forcing = forcing

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
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQ): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi, q = prognostic
        div_flux = self._compute_advection_homogeneous(PSIQ(psi, q))
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        dq = (-div_flux + fcg_drag + self.forcing) * self.masks.h
        dq_i = self._interpolate(dq)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc
        )
        div_flux = advection_psi_q
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-div_flux + fcg_drag + self.forcing) * self.masks.h
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
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_mean_flow(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_pert_i, q_pert_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_pert_i + psi_bc + self._sf_bar
        q = q_pert_i + q_bc + self._pv_bar
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc + self._pv_bar_bc
        )
        div_flux = advection_psi_q
        if self.time_stepper == "rk3":
            if self._rk3_step == 0:
                coef = 0
            elif self._rk3_step == 1:
                coef = 1
            elif self._rk3_step == 2:
                coef = 1 / 2
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
            dt = self.dt
            t = self.time.item() + coef * dt
            q_bar_t_dt = self._pv_bar_interp(t + dt)
            q_bar_t = self._pv_bar_interp(t)
            dt_q_bar = (q_bar_t_dt - q_bar_t) / dt
        else:
            dt = self.dt
            t = self.time.item()
            dt_q_bar = (self._pv_bar_interp(t + dt) - self._pv_bar) / dt
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-(div_flux + dt_q_bar) + fcg_drag + self.forcing) * self.masks.h
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
        return PSIQ(dpsi, dq)


class QGPSIQRGPsi2Transport(QGPSIQCore[PSIQTAlpha, StatePSIQAlpha]):
    """QGPSIQ with psi2 wv material derivation forcing."""

    _basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

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
        space = self.space.remove_z_h()
        self._fpsi2 = basis.localize(space.q.xy.x, space.q.xy.y)
        self._fpsi2_dx = basis.localize_dx(space.u.xy.x, space.u.xy.y)
        self._fpsi2_dy = basis.localize_dy(space.v.xy.x, space.v.xy.y)

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: Tensor,
        beta_plane: BetaPlane,
        g_prime: Tensor,
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
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            if (self.space.nl - 3) > 0
            else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)
        self._set_solver()

    def _compute_q_anom_from_psi(self, psi: Tensor) -> Tensor:
        vort = self._compute_vort_from_psi(psi)
        stretching = (
            self.beta_plane.f0**2 * (self._A11 + self.alpha * self._A12) * psi
        )
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

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self._solver_homogeneous = HomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
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

    def _set_io(self, state: StatePSIQAlpha) -> None:
        self._io = IO(state.t, state.psi, state.q, state.alpha)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StatePSIQAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl - 1,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self.alpha = torch.zeros_like(self.psi)
        self._set_io(self._state)
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

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
            torch.Tensor: -f₀²J(ѱ₁, ѱ₂)/H₂g₂
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

    def _compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQ): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi, q = prognostic
        div_flux = self._compute_advection_homogeneous(PSIQ(psi, q))
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
        dq_i = self._interpolate(dq)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc
        )
        div_flux = advection_psi_q
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (-div_flux + fcg_drag + forcing) * self.masks.h
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
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_mean_flow(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_pert_i, q_pert_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_pert_i + psi_bc + self._sf_bar
        q = q_pert_i + q_bc + self._pv_bar
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc + self._pv_bar_bc
        )
        div_flux = advection_psi_q
        if self.time_stepper == "rk3":
            if self._rk3_step == 0:
                coef = 0
            elif self._rk3_step == 1:
                coef = 1
            elif self._rk3_step == 2:
                coef = 1 / 2
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
            dt = self.dt
            t = self.time.item() + coef * dt
            q_bar_t_dt = self._pv_bar_interp(t + dt)
            q_bar_t = self._pv_bar_interp(t)
            dt_q_bar = (q_bar_t_dt - q_bar_t) / dt
        else:
            dt = self.dt
            t = self.time.item()
            dt_q_bar = (self._pv_bar_interp(t + dt) - self._pv_bar) / dt
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        forcing = self.compute_forcing(self._substep_time, psi[:, :1])
        dq = (-(div_flux + dt_q_bar) + fcg_drag + forcing) * self.masks.h
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
        return PSIQ(dpsi, dq)


class QGPSIQPsi2Transport(QGPSIQCore[PSIQTAlpha, StatePSIQAlpha]):
    """QGPSIQ with psi2 wv material derivation forcing."""

    _basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

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
        space = self.space.remove_z_h()
        self._fpsi2 = basis.localize(space.psi.xy.x, space.psi.xy.y)

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: Tensor,
        beta_plane: BetaPlane,
        g_prime: Tensor,
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
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            if (self.space.nl - 3) > 0
            else None
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)
        self._set_solver()

    def _compute_q_anom_from_psi(self, psi: Tensor) -> Tensor:
        vort = self._compute_vort_from_psi(psi)
        stretching = (
            self.beta_plane.f0**2 * (self._A11 + self.alpha * self._A12) * psi
        )
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

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self._solver_homogeneous = HomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
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

    def _set_io(self, state: StatePSIQAlpha) -> None:
        self._io = IO(state.t, state.psi, state.q, state.alpha)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StatePSIQAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl - 1,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self.alpha = torch.zeros_like(self.psi)
        self._set_io(self._state)
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

    def compute_psi_2_dt(self, time: torch.Tensor) -> torch.Tensor:
        """Compute contribution of ѱ₂'s time derivative.

        Args:
            time (torch.Tensor): Time to evaluate at.

        Returns:
            torch.Tensor: -f₀²ѱ₂/H₂g₂
        """
        dt_psi2 = self._fpsi2.dt(time)
        return (self.beta_plane.f0**2) * self._A12 * dt_psi2

    def _compute_time_derivatives_homogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for homogeneous problem.

        Args:
            prognostic (PSIQ): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi, q = prognostic
        div_flux = self._compute_advection_homogeneous(PSIQ(psi, q))
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_homogeneous(psi)
        dq = (-div_flux + fcg_drag) * self.masks.h
        dt_psi2 = self.compute_psi_2_dt(self._substep_time)
        dq_i = self._interpolate(dq) + crop(dt_psi2, 1)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_inhomogeneous(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_i, q_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_i + psi_bc
        q = q_i + q_bc
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc
        )
        div_flux = advection_psi_q
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-div_flux + fcg_drag) * self.masks.h
        dt_psi2 = self.compute_psi_2_dt(self._substep_time)
        dq_i = self._interpolate(dq) + crop(dt_psi2, 1)
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
        return PSIQ(dpsi, dq)

    def _compute_time_derivatives_mean_flow(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives for inhomogeneous problem.

        Args:
            prognostic (PSIQ): Homogeneous contribution
                of prognostic variables.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        psi_pert_i, q_pert_i = prognostic
        psi_bc, q_bc = self._solver_inhomogeneous.psiq_bc
        psi = psi_pert_i + psi_bc + self._sf_bar
        q = q_pert_i + q_bc + self._pv_bar
        advection_psi_q = self._compute_advection_inhomogeneous(
            PSIQ(psi, q), self._pv_bc + self._pv_bar_bc
        )
        div_flux = advection_psi_q
        if self.time_stepper == "rk3":
            if self._rk3_step == 0:
                coef = 0
            elif self._rk3_step == 1:
                coef = 1
            elif self._rk3_step == 2:
                coef = 1 / 2
            else:
                msg = "SSPRK3 should only perform 3 steps."
                raise ValueError(msg)
            dt = self.dt
            t = self.time.item() + coef * dt
            q_bar_t_dt = self._pv_bar_interp(t + dt)
            q_bar_t = self._pv_bar_interp(t)
            dt_q_bar = (q_bar_t_dt - q_bar_t) / dt
        else:
            dt = self.dt
            t = self.time.item()
            dt_q_bar = (self._pv_bar_interp(t + dt) - self._pv_bar) / dt
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-(div_flux + dt_q_bar) + fcg_drag) * self.masks.h
        dt_psi2 = self.compute_psi_2_dt(self._substep_time)
        dq_i = self._interpolate(dq) + crop(dt_psi2, 1)
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
        return PSIQ(dpsi, dq)

    def _compute_q_anom_from_psi(self, psi: Tensor) -> Tensor:
        vort = self._compute_vort_from_psi(psi)
        stretching = (
            self.beta_plane.f0**2 * (self._A11 + self.alpha * self._A12) * psi
        )
        if self.with_bc:
            return vort - self.masks.h * self._interpolate(stretching)
        return vort - self.masks.h * self._interpolate(
            self.masks.psi * stretching
        )


class QGPSIQPsi2TransportDR(QGPSIQPsi2Transport):
    """Mixed model using both alpha and psi2."""

    _kappa = torch.tensor(0, **defaults.get())

    @property
    def kappa(self) -> torch.Tensor:
        """Deformation radius multiplyer."""
        return self._kappa

    @kappa.setter
    def kappa(self, kappa: torch.Tensor) -> torch.Tensor:
        self._kappa = kappa
        self.compute_auxillary_matrices()
        self._set_solver()

    def compute_auxillary_matrices(self) -> None:
        """Compute auxillary matrices."""
        dtype = torch.float64
        device = DEVICE.get()
        H = self.H[:, 0, 0]
        g_prime = self.g_prime[:, 0, 0]

        H1, H2 = H
        g1, g2 = g_prime

        self.A = torch.tensor(
            [
                [
                    1 / H1 / g1 + (1 - self.kappa) / H1 / g2,
                    -(1 - self.kappa) / H1 / g2,
                ],
                [-(1 - self.kappa) / H2 / g2, (1 - self.kappa) / H2 / g2],
            ],
            dtype=dtype,
            device=device,
        )
        self._A11 = self.A[0, 0]
        self._A12 = self.A[0, 1]
