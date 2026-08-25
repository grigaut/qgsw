"""QG model with a mixed layer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.exceptions import InvalidLayerNumberError
from qgsw.fields.variables.state import (
    BaseStatePSIQSST,
    StatePSIQSST,
)
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQSST,
    PSIQSSTT,
    BasePSIQSST,
)
from qgsw.logging import getLogger
from qgsw.models.core import time_steppers
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.variable_sets import QGPSIQVariableSet
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import laplacian, nabla4
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils.interpolation import LinearInterpolation
from qgsw.utils.tensor_operations import as_singe_value_tensor

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.solver.boundary_conditions.base import Boundaries
    from qgsw.solver.pv_inversion import (
        BasePVInversion,
    )
    from qgsw.spatial.core.discretization import SpaceDiscretization2D
    from qgsw.utils.interpolation import LinearInterpolation

T = TypeVar("T", bound=BasePSIQSST)
State = TypeVar("State", bound=BaseStatePSIQSST)


logger = getLogger(__name__)


class QGPSIQSSTCore(QGPSIQCore[T, State]):
    """Finite volume multi-layer QG solver with mixed layer."""

    _H_ml = torch.tensor(100, **defaults.get())  # Mixed layer depth in meters
    _K2 = torch.tensor(380, **defaults.get())  # See Hogg, 2014
    _K4 = torch.tensor(4e10, **defaults.get())  # See Hogg, 2014
    _lambd = torch.tensor(35, **defaults.get())  # See Hogg, 2014
    _heat_cap = torch.tensor(4000, **defaults.get())  # See Kravtsov, 2022
    _rho0 = torch.tensor(1000, **defaults.get())
    _defaut_temp_atm = torch.tensor(
        282.6, **defaults.get()
    )  # See Kravtsov, 2022
    sigma = torch.tensor(5.6704e-8, **defaults.get())  # Stefan-Boltzmann
    temp_1_offset = 2
    delta_temp_1 = 8

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,
        beta_plane: BetaPlane,
        g_prime: torch.Tensor,
        optimize=True,  # noqa: ANN001
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
        self.temp_atm = self._defaut_temp_atm

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
        return self._state.sst.get() + self._sst_mean

    @property
    def sst_anom(self) -> torch.Tensor:
        """SST anomaly."""
        return self._state.sst.get()

    @property
    def sst_mean(self) -> torch.Tensor:
        """SST mean."""
        return self._sst_mean

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

    @property
    def K2(self) -> torch.Tensor:  # noqa: N802
        """Diffusion coefficient [m².s⁻¹]."""
        return self._K2

    @K2.setter
    def K2(self, value: float | torch.Tensor) -> None:  # noqa: N802
        self._K2 = as_singe_value_tensor(value)

    @property
    def K4(self) -> torch.Tensor:  # noqa: N802
        """4th order diffusion coefficient [m⁴.s⁻¹]."""
        return self._K4

    @K4.setter
    def K4(self, value: float | torch.Tensor) -> None:  # noqa: N802
        self._K4 = as_singe_value_tensor(value)

    @property
    def H_ml(self) -> torch.Tensor:  # noqa: N802
        """Mixed layer depth [m]."""
        return self._H_ml

    @H_ml.setter
    def H_ml(self, value: float | torch.Tensor) -> None:  # noqa: N802
        self._H_ml = as_singe_value_tensor(value)
        msg = f"Mixed layer depth set to {self.H_ml.item()} m"
        logger.info(msg)

    @property
    def lambd(self) -> torch.Tensor:
        """Sensible / latent heat flux coefficient [W.m⁻².K⁻¹]."""
        return self._lambd

    @lambd.setter
    def lambd(self, value: float | torch.Tensor) -> torch.Tensor:
        self._lambd = as_singe_value_tensor(value)

    @property
    def temp_atm(self) -> torch.Tensor:
        """Atmosphere temperature."""
        return self._temp_atm

    @temp_atm.setter
    def temp_atm(self, value: float | torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            value = as_singe_value_tensor(value)
        if value.numel() == 1:
            self._temp_atm = (
                torch.ones_like(
                    self.space.h.xyh.x[:1].tile(self.n_ens, 1, 1, 1)
                )
                * value.squeeze()
            )
            return
        if value.shape != (s := self.q[:, :1, ...].shape):
            msg = f"Atmosphere temperature field should be {s}-shaped."
            raise ValueError(msg)
        self._temp_atm = value

    @property
    def rho0(self) -> torch.Tensor:
        """Reference density [kg.m⁻³]."""
        return self._rho0

    @rho0.setter
    def rho0(self, value: float | torch.Tensor) -> None:
        self._rho0 = as_singe_value_tensor(value)

    @property
    def heat_cap(self) -> torch.Tensor:
        """Heat capacity [J.kg⁻¹.K⁻¹]."""
        return self._heat_cap

    @heat_cap.setter
    def heat_cap(self, value: float | torch.Tensor) -> None:
        self._heat_cap = as_singe_value_tensor(value)

    def _set_io(self, state: State) -> None:
        self._io = IO(state.t, state.psi, state.q, state.sst)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StatePSIQSST.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self._sst_mean = (self._state.sst.get() * self.masks.h).mean()
        self._state.update_sst(self._state.sst.get() - self._sst_mean)
        self._set_io(self._state)
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

    def set_mean_flow(
        self,
        sf_bar_interp: LinearInterpolation[torch.Tensor],
        pv_bar_interp: LinearInterpolation[torch.Tensor],
        sf_bar_bc_interp: LinearInterpolation[Boundaries],
        pv_bar_bc_interp: LinearInterpolation[Boundaries],
    ) -> None:
        """Not implemented."""
        raise NotImplementedError

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
        sst_bc: Boundaries = self._sst_bc_interp(time) - self._sst_mean
        if self.wide:
            if pv_bc.width != 3 or sst_bc.width != 3:
                msg = (
                    "For wide boundary, pv_bc and sst_bc must"
                    " be 3 points wide."
                )
                raise ValueError(msg)
            self._pv_bc = pv_bc
            self._sst_bc = sst_bc
            if self._with_mean_flow:
                self._pv_bar = self._pv_bar_interp(time)
                self._pv_bar_bc = self._pv_bar_bc_interp(time)
                self._pv_bc -= self._pv_bar_bc

        else:
            self._pv_bc = pv_bc.get_band(0)
            self._sst_bc = sst_bc.get_band(0)
            if self._with_mean_flow:
                self._pv_bar = self._pv_bar_interp(time)
                self._pv_bar_bc = self._pv_bar_bc_interp(time).get_band(0)
                self._pv_bc -= self._pv_bar_bc

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

    def compute_fluxes(
        self,
        sst_anom: torch.Tensor,
        *,
        with_atm_convective: bool = False,
        with_radiative: bool = False,
    ) -> torch.Tensor:
        """Compute fluxes.

        Args:
            sst_anom (torch.Tensor): SST anomaly field.
            with_atm_convective (bool, optional): Whether to include
                atmospheric convective fluxes. Defaults to False.
            with_radiative (bool, optional): Whether to include radiative
                fluxes. Defaults to False.

        Returns:
            torch.Tensor: Fluxes.
        """
        forcing = torch.zeros_like(sst_anom)
        factor = 1 / self.rho0 / self.heat_cap / self.H_ml
        sst = sst_anom + self._sst_mean
        if with_atm_convective:
            forcing += -self.lambd * (sst - self.temp_atm)
        if with_radiative:
            forcing += self.sigma * ((self.temp_atm) ** 4 - (sst) ** 4)
        return forcing * factor

    def compute_diffusion(
        self,
        sst_anom: torch.Tensor,
        sst_anom_bcs: Boundaries | None = None,
        *,
        with_2nd_order: bool = True,
        with_4th_order: bool = True,
    ) -> torch.Tensor:
        """Compute diffusion.

        Args:
            sst_anom (torch.Tensor): SST anomaly field.
            sst_anom_bcs (torch.Tensor | None): SST anomaly boundaries.
                If None, padding will be done through boundray replication.
                Defaults to None.
            with_2nd_order (bool, optional): Whether to include 2nd order
                diffusion. Defaults to True.
            with_4th_order (bool, optional): Whether to include 4th order
                diffusion. Defaults to True.

        Returns:
            torch.Tensor: _description_
        """
        diffusion = torch.zeros_like(sst_anom)
        dx, dy = self.space.dx, self.space.dy
        if with_2nd_order:
            if sst_anom_bcs is None:
                padded = F.pad(sst_anom, (1, 1, 1, 1), mode="replicate")
            else:
                padded = sst_anom_bcs.get_band(0).expand(sst_anom)
            diffusion += laplacian(padded, dx, dy) * self.K2
        if with_4th_order:
            if sst_anom_bcs is None:
                padded = F.pad(sst_anom, (2, 2, 2, 2), mode="replicate")
            else:
                padded_in = sst_anom_bcs.get_band(0).expand(sst_anom)
                if sst_anom_bcs.width >= 2:
                    padded = sst_anom_bcs.get_band(1).expand(padded_in)
                else:
                    padded = F.pad(padded_in, (1, 1, 1, 1), mode="replicate")
            diffusion -= nabla4(padded, dx, dy) * self.K4
        return diffusion

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
            + self.beta_plane.f0 / self.H * (e[:, :-1] - e[:, 1:])
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

        temp_1_anom = torch.mean(sst_anom) - self.temp_1_offset

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
            + self.beta_plane.f0 / self.H * (e[:, :-1] - e[:, 1:])
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

        temp_1_anom = torch.mean(sst_anom) - self.temp_1_offset

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
        self._sst_mean = torch.mean(sst * self.masks.h)
        self._state.update_sst(sst - self._sst_mean)

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
        prognostic_i = PSIQSST(psi - psi_bc, q, sst)
        psiqsst_i = self._timestep(prognostic_i)
        self._set_boundaries(self.time.item())
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        return PSIQSST(psiqsst_i.psi + psi_bc, psiqsst_i.q, psiqsst_i.sst)

    @torch.enable_grad()
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.update_psiqsst(self.update(self._state.prognostic.psiqsst))
        sst_anom = self.sst_anom
        temp_1_anom = torch.mean(sst_anom * self.masks.h) - self.temp_1_offset
        sst_anom = torch.where(sst_anom >= temp_1_anom, sst_anom, temp_1_anom)
        self._state.update_sst(sst_anom)

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


class QGPSIQSST(QGPSIQSSTCore[PSIQSSTT, StatePSIQSST]):
    """Quasi Geostrophic Model with mixed layer and SST."""

    _type = ModelName.QUASI_GEOSTROPHIC_ML
