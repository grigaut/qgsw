"""Collinear & filtered QGPSIQ model."""

import torch
from torch import Tensor

from qgsw.fields.variables.state import StatePSIQAlpha
from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQTAlpha,
)
from qgsw.filters.base import _Filter
from qgsw.filters.high_pass import GaussianHighPass2D
from qgsw.models.io import IO
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.finite_diff import laplacian, laplacian_h
from qgsw.solver.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.solver.pv_inversion import (
    HomogeneousPVInversionCollinear,
    InhomogeneousPVInversionCollinear,
)
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import DEVICE, defaults


class QGPSIQCollinearFilteredSF(QGPSIQCore[PSIQTAlpha, StatePSIQAlpha]):
    """QGPSIQ model with collinearity coefficient and filtered sf."""

    _sigma = 1

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: Tensor,  # noqa: N803
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
        self._g2 = g_prime[1:2]
        g_tilde = g_prime[:1] * g_prime[1:2] / (g_prime[1:2] + g_prime[:1])
        self._H1 = H[:1]
        super().__init__(
            space_2d=space_2d,
            H=H[:1],
            beta_plane=beta_plane,
            g_prime=g_tilde,
            optimize=optimize,
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)

    @property
    def sigma(self) -> torch.Tensor:
        """Filter standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        self._sigma = sigma
        self._filter = GaussianHighPass2D(self.sigma)

    @property
    def filter(self) -> _Filter:
        """2D filter."""
        return self._filter

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
        self.homsol_mean = (interpolate(self.homsol) * self.masks.h).mean(
            (-1, -2),
            keepdim=True,
        )
        self.helmholtz_dst = self.helmholtz_dst.type(torch.float32)

    def _set_io(self, state: StatePSIQAlpha) -> None:
        self._io = IO(state.t, state.psi, state.q, state.alpha)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StatePSIQAlpha.steady(
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
        psi_filt = self._filter(psi[0, 0])
        psi_filt = psi_filt.unsqueeze(0).unsqueeze(0)
        psi2 = self.alpha * psi_filt
        source_term = -1 * self.beta_plane.f0**2 * psi2 / self._H1 / self._g2
        beta_effect = self.beta_plane.beta * (self._y - self._y0)
        return self.masks.h * (
            self._interpolate(
                self.masks.psi * (lap_psi + source_term - stretching),
            )
            + beta_effect
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
        psi_i = torch.zeros(
            (1, 1, q.shape[-2] + 2, q.shape[-1] + 2),
            **defaults.get(),
        )
        for _ in range(100):
            psi_filt = self._filter(psi_i[0, 0])
            psi_filt = psi_filt.unsqueeze(0).unsqueeze(0)
            psi2 = self.alpha * psi_filt
            psi2_interp = self._interpolate(
                self._interpolate(psi2),
            )
            helmholtz_rhs = torch.einsum(
                "lm,...mxy->...lxy",
                self.Cl2m,
                q + self.beta_plane.f0**2 * psi2_interp / self._H1 / self._g2,
            )
            if self.cap_matrices is not None:
                msg = "Not implemented for non-rectangular geometries."
                raise NotImplementedError(msg)
            psi_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dst)
            if torch.isclose(psi_i, psi_modes).all():
                break
            psi_i = psi_modes

        # Add homogeneous solutions to ensure mass conservation
        gamma = (
            -self._interpolate(psi_modes).mean((-1, -2), keepdim=True)
            / self.homsol_mean
        )
        psi_modes += gamma * self.homsol
        return torch.einsum("lm,...mxy->...lxy", self.Cm2l, psi_modes)

    def compute_auxillary_matrices(self) -> None:
        """Compute auxiliary matrices."""
        super().compute_auxillary_matrices()

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


class QGPSIQCollinearSF(QGPSIQCore[PSIQTAlpha, StatePSIQAlpha]):
    """Collinear QGPSIQ model."""

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: Tensor,  # noqa: N803
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
        self.compute_auxillary_matrices()
        self._set_solver()
        if self._with_bc:
            sf_bc = self._sf_bc_interp(self.time.item())
            if self._with_mean_flow:
                sf_bar_bc = self._sf_bar_bc_interp(self.time.item())
                self._solver_inhomogeneous.set_boundaries(
                    sf_bc.get_band(0) - sf_bar_bc.get_band(0)
                )
            else:
                self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

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
        elif self.space.nl - 1 == 2:  # noqa: PLR2004
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
        elif self.space.nl - 1 == 2:  # noqa: PLR2004
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
