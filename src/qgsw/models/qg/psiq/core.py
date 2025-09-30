"""Usual QG Model."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from qgsw import verbose
from qgsw.exceptions import InvalidLayerNumberError, UnsetStencilError
from qgsw.fields.variables.state import BaseStatePSIQ, StatePSIQ
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQT,
    BasePSIQ,
)
from qgsw.masks import Masks
from qgsw.models.base import _Model
from qgsw.models.core import time_steppers
from qgsw.models.core.flux import (
    div_flux_3pts,
    div_flux_3pts_mask,
    div_flux_5pts,
    div_flux_5pts_mask,
    div_flux_5pts_only,
    div_flux_5pts_with_bc,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.psiq.variable_sets import QGPSIQVariableSet
from qgsw.models.qg.stretching_matrix import (
    compute_A,
)
from qgsw.solver.finite_diff import grad_perp, laplacian, laplacian_h
from qgsw.solver.pv_inversion import (
    BasePVInversion,
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.solver.boundary_conditions.base import Boundaries
    from qgsw.spatial.core.discretization import SpaceDiscretization2D
    from qgsw.utils.interpolation import LinearInterpolation

T = TypeVar("T", bound=BasePSIQ)
State = TypeVar("State", bound=BaseStatePSIQ)


class QGPSIQCore(_Model[T, State, PSIQ], Generic[T, State]):
    """Finite volume multi-layer QG solver."""

    _flux_stencil = 5

    _time_stepper: str = "rk3"
    wide = False

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
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
        # physical params
        self._switch_to_homogeneous()
        _Model.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=True,
        )

        # grid params
        self.y0 = 0.5 * self._space.ly

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # initialize state variables
        self._set_utils(optimize)
        self._optim = optimize

        self.zeros_inside = (
            torch.zeros(
                (self.n_ens, self.space.nl - 2, self.space.nx, self.space.ny),
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            if (self.space.nl - 2) > 0
            else None
        )
        # wind forcing
        self.set_wind_forcing(0.0, 0.0)

    @property
    def y0(self) -> float:
        """Reference y0."""
        return self._y0

    @y0.setter
    def y0(self, y0: float) -> None:
        """Set y0."""
        self._y0 = y0
        # Beta effect
        y = self.space.q.xyz.y[0, 0, :].unsqueeze(0)
        beta_effect = self.beta_plane.beta * (y - self.y0)
        if hasattr(self, "_state"):
            verbose.display(
                "WARNING: The β-effect has been modified."
                "The potential vorticity might be inconsistent with actual"
                "β-effect value. Best would be setting y0 "
                "right after instantiation.",
                trigger_level=1,
            )
        self._beta_effect = beta_effect

    @property
    def flux_stencil(self) -> int:
        """Flux stencil size."""
        try:
            return self._flux_stencil
        except AttributeError as e:
            msg = "Please set the flux stencil."
            raise UnsetStencilError(msg) from e

    @flux_stencil.setter
    def flux_stencil(self, stencil: int) -> None:
        stencil = int(stencil)
        if stencil not in (3, 5):
            msg = "The stencil can only be 3 or 5."
            raise ValueError(msg)
        self._flux_stencil = stencil
        self._set_flux()

    @property
    def masks(self) -> Masks:
        """Masks."""
        try:
            return self._masks
        except AttributeError:
            self.masks = Masks.empty_tensor(
                self.space.nx,
                self.space.ny,
                device=DEVICE.get(),
            )
            return self._masks

    @masks.setter
    def masks(self, mask: torch.Tensor) -> None:
        """Masks setter."""
        ModelParamChecker.masks.fset(self, mask)
        # Set state
        if hasattr(self, "_state"):
            verbose.display(
                "WARNING: The masks have been modified."
                " The stream function and potential have"
                " therefore been set to 0.",
                trigger_level=1,
            )
        self._set_state()
        self._create_diagnostic_vars(self._state)
        # Set solver
        self._set_solver()
        # flux computations
        with contextlib.suppress(UnsetStencilError):
            self._set_flux()

    @property
    def psi(self) -> torch.Tensor:
        """StatePSIQ Variable psi: Stream function.

        └── (n_ens, nl, nx+1,ny+1)-shaped.
        """
        return self._state.psi.get()

    @property
    def q(self) -> torch.Tensor:
        """StatePSIQ Variable q: Potential Vorticity.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self._state.q.get()

    @property
    def q_anom(self) -> torch.Tensor:
        """Potential Vorticity anomaly.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self.q - self._beta_effect

    @property
    def vorticity(self) -> torch.Tensor:
        """Vorticity.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self._compute_vort_from_psi(self.psi)

    @property
    def time_stepper(self) -> str:
        """Time stepper."""
        return self._time_stepper

    @time_stepper.setter
    def time_stepper(self, time_stepper: str) -> None:
        """Time stepper setter."""
        time_steppers.validate(time_stepper)
        self._time_stepper = time_stepper
        verbose.display(
            "Time stepper set to: " + time_stepper,
            trigger_level=1,
        )

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

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self._solver_homogeneous = HomogeneousPVInversion(
            self.A,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversion(
            self.A,
            self._beta_plane.f0,
            self.space.dx,
            self.space.dy,
            self._masks,
        )

    def _set_flux(self) -> None:
        """Set the fluxes utils."""
        if self.with_bc:
            return self._set_flux_inhomogeneous()
        return self._set_flux_homogeneous()

    def _set_flux_homogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        if self.flux_stencil == 5:  # noqa: PLR2004
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = lambda q, u, v: div_flux_5pts_mask(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                    self.masks.u_distbound1[..., 1:-1, :],
                    self.masks.u_distbound2[..., 1:-1, :],
                    self.masks.u_distbound3plus[..., 1:-1, :],
                    self.masks.v_distbound1[..., 1:-1],
                    self.masks.v_distbound2[..., 1:-1],
                    self.masks.v_distbound3plus[..., 1:-1],
                )
            else:
                div_flux = lambda q, u, v: div_flux_5pts(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
        elif self.flux_stencil == 3:  # noqa: PLR2004
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = lambda q, u, v: div_flux_3pts_mask(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                    self.masks.u_distbound1[..., 1:-1, :],
                    self.masks.u_distbound2plus[..., 1:-1, :],
                    self.masks.v_distbound1[..., 1:-1],
                    self.masks.v_distbound2plus[..., 1:-1],
                )
            else:
                div_flux = lambda q, u, v: div_flux_3pts(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
        else:
            msg = f"Invalid stencil value: {self.flux_stencil}"
            raise ValueError(msg)

        self.div_flux = (
            OptimizableFunction(div_flux) if self._optim else div_flux
        )

    def _set_flux_inhomogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        if self.flux_stencil == 5:  # noqa: PLR2004
            if len(self.masks.psi_irrbound_xids) > 0:
                msg = (
                    "Inhomogeneous pv reconstruction not "
                    "implemented for non-regular geometry."
                )
                raise NotImplementedError(msg)
            if self.wide:
                div_flux = lambda q, u, v: div_flux_5pts_only(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
            else:
                div_flux = lambda q, u, v: div_flux_5pts_with_bc(
                    q,
                    u,
                    v,
                    self.space.dx,
                    self.space.dy,
                )
        elif self.flux_stencil == 3:  # noqa: PLR2004
            msg = (
                "Inhomogeneous pv reconstruction not "
                "implemented for 3 pts stencil."
            )
            raise NotImplementedError(msg)
        else:
            msg = f"Invalid stencil value: {self.flux_stencil}"
            raise ValueError(msg)

        self.div_flux = (
            OptimizableFunction(div_flux) if self._optim else div_flux
        )

    def _switch_to_inhomogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        if self.with_bc:
            return
        self._with_bc = True
        self._set_flux()

    def _switch_to_homogeneous(self) -> None:
        """Switch to an inhomogeneous solver."""
        self._with_bc = False

    def _set_io(self, state: StatePSIQ) -> None:
        self._io = IO(state.t, state.psi, state.q)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StatePSIQ.steady(
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

    def _set_utils(self, optimize: bool) -> None:  # noqa: FBT001
        """Set utils.

        Args:
            optimize (bool): Whether to optimize or not.
        """
        if optimize:
            self._grad_perp = OptimizableFunction(grad_perp)
            self._points_to_surfaces = OptimizableFunction(points_to_surfaces)
            self._laplacian_h = OptimizableFunction(laplacian_h)
        else:
            self._grad_perp = grad_perp
            self._points_to_surfaces = points_to_surfaces
            self._laplacian_h = laplacian_h

    def compute_auxillary_matrices(self) -> None:
        """Compute auxiliary matrix."""
        # A operator
        self.A = compute_A(
            self.H[:, 0, 0],
            self.g_prime[:, 0, 0],
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    def _compute_q_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute stream function from stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return (
            self._compute_q_anom_from_psi(psi)
            + self.masks.h * self._beta_effect
        )

    def _compute_q_anom_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute potential vorticity anomaly from stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Potential vorticity anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        vort = self._compute_vort_from_psi(psi)
        stretching = self.beta_plane.f0**2 * torch.einsum(
            "lm,...mxy->...lxy",
            self.A,
            psi,
        )
        if self.with_bc:
            return vort - self.masks.h * self._points_to_surfaces(stretching)
        return vort - self.masks.h * self._points_to_surfaces(
            self.masks.psi * stretching
        )

    def _compute_vort_from_psi(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute vorticity from streamfunction.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            boundary = self._sf_bc_interp(self.time.item())
            psi_with_bc = boundary.get_band(1).expand(psi)
            self.p = psi_with_bc
            lap_psi = laplacian(psi_with_bc, self.space.dx, self.space.dy)
            return self.masks.h * self._points_to_surfaces(lap_psi)
        lap_psi = laplacian_h(psi, self.space.dx, self.space.dy)
        return self.masks.h * (
            self._points_to_surfaces(
                self.masks.psi * (lap_psi),
            )
        )

    def _compute_psi_from_q(self, q: torch.Tensor) -> torch.Tensor:
        """Compute stream function from potential vorticity.

        Args:
            q (torch.Tensor): Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        return self.solver.compute_stream_function(self._points_to_surfaces(q))

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
        self._curl_tau = curl_tau.unsqueeze(0).unsqueeze(0) / self.H[0]

    def set_boundary_maps(
        self,
        sf_bc_interp: LinearInterpolation[Boundaries],
        pv_bc_interp: LinearInterpolation[Boundaries],
    ) -> None:
        """Set the boundary maps.

        Args:
            sf_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for stream function at locations
                (imin,imax+1,jmin,jmax+1).
            pv_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for potential vorticity at locations
                (imin,imax,jmin,jmax).
        """
        self._switch_to_inhomogeneous()
        self._sf_bc_interp = sf_bc_interp
        self._pv_bc_interp = pv_bc_interp
        self._set_boundaries(self.time.item())

    def _set_boundaries(self, time: float) -> None:
        """Set the boundaries to match given time.

        Args:
            time (float): Time.
        """
        sf_boundary = self._sf_bc_interp(time)
        pv_boundary = self._pv_bc_interp(time)
        self._solver_inhomogeneous.set_boundaries(sf_boundary.get_band(0))

        if self.wide:
            if pv_boundary.width != 3:  # noqa: PLR2004
                msg = "For wide boundary, pv_boundary must be 3 points wide."
                raise ValueError(msg)
            q_ext = torch.zeros_like(self.q)
            self._q_ext = pv_boundary.expand(q_ext)
        else:
            q_ext = torch.zeros_like(self.q)
            self._q_ext = pv_boundary.get_band(0).expand(q_ext)

    def _compute_advection_homogeneous(self, psiq: PSIQ) -> torch.Tensor:
        """Compute advection pv advection for homogeneous problem.

        Args:
            psiq (PSIQ): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: RHS: J(ѱ, q)
                └──  (n_ens, nl, nx, ny)-shaped
        """
        psi, q = psiq
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        return self.div_flux(
            q,
            u[..., 1:-1, :],
            v[..., 1:-1],
        )

    def _compute_advection_inhomogeneous(self, psiq: PSIQ) -> torch.Tensor:
        """Compute advection pv advection for inhomogeneous problem.

        Args:
            psiq (PSIQ): Prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: RHS: J(ѱ, q)
                └──  (n_ens, nl, nx, ny)-shaped
        """
        psi, q = psiq
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        q_full = self._q_ext
        if self.wide:
            q_full[..., 3:-3, 3:-3] = q
        else:
            q_full[..., 1:-1, 1:-1] = q
        return self.div_flux(q_full, u, v)

    def _compute_drag_homogeneous(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute wind and bottom drag contribution.

        Args:
            psi (torch.Tensor): Stream function.
                └──  psi: (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Wind and bottom drag.
                └──  (n_ens, nl, nx, ny)-shaped
        """
        omega = self._points_to_surfaces(
            self._laplacian_h(psi, self.space.dx, self.space.dy)
            * self.masks.psi,
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl == 1:
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl == 2:  # noqa: PLR2004
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
        omega = points_to_surfaces(
            laplacian(sf_wide, self.space.dx, self.space.dy)
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.space.nl == 1:
            fcg_drag = self._curl_tau + bottom_drag
        elif self.space.nl == 2:  # noqa: PLR2004
            fcg_drag = torch.cat([self._curl_tau, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self._curl_tau, self.zeros_inside, bottom_drag],
                dim=-3,
            )
        return fcg_drag

    def compute_time_derivatives(self, prognostic: PSIQ) -> PSIQ:
        """Compute time derivatives.

        Args:
            prognostic (PSIQ): prognostic tuple.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        if self.with_bc:
            return self._compute_time_derivatives_inhomogeneous(prognostic)
        return self._compute_time_derivatives_homogeneous(prognostic)

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
        dq_i = self._points_to_surfaces(dq)
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )
        self.dpsi = dpsi
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
        advection_psi_q = self._compute_advection_inhomogeneous(PSIQ(psi, q))
        div_flux = advection_psi_q
        # wind forcing + bottom drag
        fcg_drag = self._compute_drag_inhomogeneous(psi)
        dq = (-div_flux + fcg_drag) * self.masks.h
        dq_i = self._points_to_surfaces(dq)
        self.dqi = dq_i
        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )
        if self.time_stepper == "rk3":
            self._rk3_step += 1
            self._set_boundaries(
                self.time.item() + self._rk3_step / 3 * self.dt
            )

        return PSIQ(dpsi, dq)

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
        psi = self.solver.compute_stream_function(
            self._points_to_surfaces(q_anom)
        )
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

    def set_psiq(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set both psi and q.

        Args:
            psi (torch.Tensor): Stream function tensor.
            q (torch.Tensor): Potential vorticity tensor.
        """
        self._state.update_psiq(PSIQ(psi, q))

    def set_psiqanom(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set both psi and q.

        Args:
            psi (torch.Tensor): Stream function tensor.
            q (torch.Tensor): Potential vorticity tensor.
        """
        self._state.update_psiq(
            PSIQ(psi, q + self._masks.h * self._beta_effect)
        )

    def update(self, prognostic: PSIQ) -> PSIQ:
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
            return self._update_inhomogeneous(prognostic)
        return self._update_homogeneous(prognostic)

    def _timestep(self, prognostic: PSIQ) -> PSIQ:
        if self.time_stepper == "rk3":
            self._rk3_step = 0
            psiq = time_steppers.rk3_ssp(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        elif self.time_stepper == "euler":
            psiq = time_steppers.euler(
                prog=prognostic,
                dt=self.dt,
                time_derivation_func=self.compute_time_derivatives,
            )
        else:
            msg = f"Invalid time stepper: {self.time_stepper}"
            raise ValueError(msg)
        self._state.increment_time(self.dt)
        return psiq

    def _update_homogeneous(self, prognostic: PSIQ) -> PSIQ:
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

    def _update_inhomogeneous(self, prognostic: PSIQ) -> PSIQ:
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
        prognostic_i = PSIQ(prognostic.psi - psi_bc, prognostic.q)
        psiq_i = self._timestep(prognostic_i)
        self._set_boundaries(self.time.item())
        psi_bc = self._solver_inhomogeneous.psiq_bc.psi
        return PSIQ(psiq_i.psi + psi_bc, psiq_i.q)

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.update_psiq(self.update(self._state.prognostic.psiq))

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


class QGPSIQ(QGPSIQCore[PSIQT, StatePSIQ]):
    """Usual Quasi Geostrophic Model (psi / pv formulation)."""

    _type = ModelName.QUASI_GEOSTROPHIC_USUAL
