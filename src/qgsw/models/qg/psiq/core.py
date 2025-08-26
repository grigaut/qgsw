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
from qgsw.models.core import schemes
from qgsw.models.core.flux import (
    div_flux_3pts,
    div_flux_3pts_mask,
    div_flux_5pts,
    div_flux_5pts_mask,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.psiq.variable_sets import QGPSIQVariableSet
from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_layers_to_mode_decomposition,
)
from qgsw.solver.finite_diff import grad_perp, laplacian_h
from qgsw.solver.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D

T = TypeVar("T", bound=BasePSIQ)
State = TypeVar("State", bound=BaseStatePSIQ)


class QGPSIQCore(_Model[T, State, PSIQ], Generic[T, State]):
    """Finite volume multi-layer QG solver."""

    _flux_stencil = 5

    dtype = torch.float64
    device = DEVICE

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
        _Model.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=True,
        )

        # grid params
        self._y = torch.linspace(
            0.5 * self._space.dy,
            self.space.ly - 0.5 * self._space.dy,
            self._space.ny,
            dtype=torch.float64,
            device=DEVICE.get(),
        ).unsqueeze(0)
        self._y0 = 0.5 * self._space.ly

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

        # Beta-effect
        self._beta_effect = self.beta_plane.beta * (self._y - self._y0)

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

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # homogeneous Helmholtz solutions
        cst = torch.ones(
            (1, self.space.nl, self.space.nx + 1, self.space.ny + 1),
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        if len(self.masks.psi_irrbound_xids) > 0:
            self.cap_matrices = compute_capacitance_matrices(
                self.helmholtz_dst,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
            )
            sol = solve_helmholtz_dstI_cmm(
                (cst * self.masks.psi)[..., 1:-1, 1:-1],
                self.helmholtz_dst,
                self.cap_matrices,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
                self.masks.psi,
            )
        else:
            self.cap_matrices = None
            sol = solve_helmholtz_dstI(
                cst[..., 1:-1, 1:-1],
                self.helmholtz_dst,
            )

        self.homsol = cst + sol * self.beta_plane.f0**2 * self.lambd
        self.homsol_mean = (
            points_to_surfaces(self.homsol) * self.masks.h
        ).mean(
            (-1, -2),
            keepdim=True,
        )
        self.helmholtz_dst = self.helmholtz_dst.type(torch.float32)

    def _set_flux(self) -> None:
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

        # layer-to-mode and mode-to-layer matrices
        self.Cm2l, lambd, self.Cl2m = compute_layers_to_mode_decomposition(
            self.A,
        )
        self.lambd = lambd.reshape((1, self.space.nl, 1, 1))

        # For Helmholtz equations
        nx, ny = self.space.nx, self.space.ny
        laplace_dst = (
            compute_laplace_dstI(
                nx,
                ny,
                self.space.dx,
                self.space.dy,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.helmholtz_dst = laplace_dst - self.beta_plane.f0**2 * self.lambd

    def _compute_q_from_psi(self, psi: torch.Tensor) -> None:
        """Compute stream function from stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        lap_psi = laplacian_h(psi, self.space.dx, self.space.dy)
        stretching = self.beta_plane.f0**2 * torch.einsum(
            "lm,...mxy->...lxy",
            self.A,
            psi,
        )
        return self.masks.h * (
            self._points_to_surfaces(
                self.masks.psi * (lap_psi - stretching),
            )
            + self._beta_effect
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
        helmholtz_rhs = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cl2m,
            q,
        )
        if self.cap_matrices is not None:
            psi_modes = solve_helmholtz_dstI_cmm(
                helmholtz_rhs * self.masks.psi[..., 1:-1, 1:-1],
                self.helmholtz_dst,
                self.cap_matrices,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
                self.masks.psi,
            )
        else:
            psi_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dst)

        # Add homogeneous solutions to ensure mass conservation
        alpha = (
            -self._points_to_surfaces(psi_modes).mean((-1, -2), keepdim=True)
            / self.homsol_mean
        )
        psi_modes += alpha * self.homsol
        return torch.einsum("lm,...mxy->...lxy", self.Cm2l, psi_modes)

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

    def advection_rhs(self, prognostic: PSIQ) -> torch.Tensor:
        """Right hand side advection.

        Args:
            prognostic (PSIQ): Prognostic psi and q.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  q : (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: dq.
                └── dq: (n_ens, nl, nx, ny)-shaped
        """
        psi, q = prognostic
        u, v = self._grad_perp(psi)
        u /= self.space.dy
        v /= self.space.dx
        div_flux = self.div_flux(
            q,
            u[..., 1:-1, :],
            v[..., 1:-1],
        )

        # wind forcing + bottom drag
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

        return (-div_flux + fcg_drag) * self.masks.h

    def compute_time_derivatives(
        self,
        prognostic: PSIQ,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time derivatives.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dpsi, dq
                ├── dpsi: (n_ens, nl, nx+1, ny+1)-shaped
                └──  dq : (n_ens, nl, nx, ny)-shaped
        """
        dq = self.advection_rhs(prognostic)

        # Solve Helmholtz equation
        dq_i = self._points_to_surfaces(dq)
        dpsi = self._compute_psi_from_q(dq_i)

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

        Args:
            q (torch.Tensor): Potential vorticity.
                └── (n_ens, nl, nx, ny)-shaped
        """
        self.set_q_anomaly(q_anom=q - self._beta_effect)

    def set_q_anomaly(self, q_anom: torch.Tensor) -> None:
        """Set the value of potential vorticity.

        Args:
            q_anom (torch.Tensor): Potential vorticity anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        q_anom_i = self._points_to_surfaces(q_anom)
        psi = self._compute_psi_from_q(q_anom_i)
        self._state.update_psiq(PSIQ(psi, q_anom + self._beta_effect))

    def set_psi(self, psi: torch.Tensor) -> None:
        """Set the value of stream function.

        Args:
            psi (torch.Tensor): Stream function.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        q = self._compute_q_from_psi(psi)
        self._state.update_psiq(PSIQ(psi, q))

    def update(self, prognostic: PSIQ) -> PSIQ:
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
        return schemes.rk3_ssp(
            prog=prognostic,
            dt=self.dt,
            time_derivation_func=self.compute_time_derivatives,
        )

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        super().step()
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
