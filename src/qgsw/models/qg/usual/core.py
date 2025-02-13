"""Usual QG Model."""

import torch

from qgsw.fields.variables.prognostic_tuples import PSIQ, PSIQT
from qgsw.fields.variables.state import StatePSIQ
from qgsw.models.base import _Model
from qgsw.models.core import schemes
from qgsw.models.core.finite_diff import grad_perp, laplacian_h
from qgsw.models.core.flux import (
    div_flux_3pts,
    div_flux_3pts_mask,
    div_flux_5pts,
    div_flux_5pts_mask,
)
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.io import IO
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_layers_to_mode_decomposition,
)
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE


class QGPSIQ(_Model[PSIQT, StatePSIQ, PSIQ]):
    """Finite volume multi-layer QG solver."""

    flux_stencil = 5
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
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
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
        self._set_state()

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
        self.wind_forcing = torch.zeros(
            (1, 1, self.space.nx, self.space.ny),
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @ModelParamChecker.masks.setter
    def masks(self, mask: torch.Tensor) -> None:
        """Masks setter."""
        ModelParamChecker.masks.fset(self, mask)
        # flux computations
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

        self.div_flux = OptimizableFunction(div_flux) if True else div_flux

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
        self._io = IO(
            t=self._state.t,
            psi=self._state.psi,
            q=self._state.q,
        )
        q = self._compute_q_from_psi(self.psi)
        self._state.update_psiq(PSIQ(self.psi, q))

    def _set_utils(self, optimize: bool) -> None:  # noqa: FBT001
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
        """Compute potential vorticity from stream function."""
        lap_psi = laplacian_h(psi, self.space.dx, self.space.dy)
        stretching = self.beta_plane.f0**2 * torch.einsum(
            "lm,...mxy->...lxy",
            self.A,
            psi,
        )
        beta_effect = self.beta_plane.beta * (self._y - self._y0)
        return self.masks.h * (
            self._points_to_surfaces(
                self.masks.psi * (lap_psi - stretching),
            )
            + beta_effect
        )

    def _compute_psi_from_q(self, q_rhs: torch.Tensor) -> torch.Tensor:
        helmholtz_rhs = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cl2m,
            q_rhs,
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

    def set_wind_forcing(self, curl_tau: torch.Tensor) -> None:
        """Set the wind forcing.

        Args:
            curl_tau (torch.Tensor): Wind curl.
        """
        self.wind_forcing = curl_tau / self.H[0]

    def advection_rhs(self, prognostic: PSIQ) -> torch.Tensor:
        """Right hand side advection."""
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
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.space.nl == 2:  # noqa: PLR2004
            fcg_drag = torch.cat([self.wind_forcing, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [self.wind_forcing, self.zeros_inside, bottom_drag],
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
        """
        dq = self.advection_rhs(prognostic)

        # Solve Helmholtz equation
        dq_i = self._points_to_surfaces(dq)
        dpsi = self._compute_psi_from_q(dq_i)

        return PSIQ(dpsi, dq)

    def set_q(self, q: torch.Tensor) -> None:
        """Set the value of potential vorticity.

        Args:
            q (torch.Tensor): Potential vorticity.
        """
        q_i = self._points_to_surfaces(q)
        psi = self._compute_psi_from_q(q_i)
        self._state.update_psiq(PSIQ(psi, q))

    def set_psi(self, psi: torch.Tensor) -> None:
        """Set the value of stream function.

        Args:
            psi (torch.Tensor): Stream function.
        """
        q = self._compute_q_from_psi(psi)
        self._state.update_psiq(PSIQ(psi, q))

    def update(self, prognostic: PSIQ) -> PSIQ:
        """Update prognostic tuple.

        Args:
            prognostic (PSIQ): Prognostic variable to advect.

        Returns:
            PSIQ: Updated prognostic variable to advect.
        """
        return schemes.rk3_ssp(
            prog=prognostic,
            dt=self.dt,
            time_derivation_func=self.compute_time_derivatives,
        )

    def step(self) -> None:
        """Time itegration with SSP-RK3 scheme."""
        """Performs one step time-integration with RK3-SSP scheme."""
        super().step()
        self._state.update_psiq(self.update(self._state.prognostic.psiq))
