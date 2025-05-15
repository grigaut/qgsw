"""Collinear & filtered QGPSIQ model."""

import warnings
from functools import cached_property

import torch
from torch import Tensor

from qgsw.fields.variables.prognostic_tuples import (
    PSIQ,
    PSIQTAlpha,
)
from qgsw.fields.variables.state import StatePSIQAlpha
from qgsw.filters.base import _Filter
from qgsw.filters.high_pass import GaussianHighPass2D
from qgsw.models.core.finite_diff import laplacian_h
from qgsw.models.core.helmholtz import solve_helmholtz_dstI
from qgsw.models.io import IO
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import defaults


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
        self._H1 = H[1:2]
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

    @cached_property
    def offset_psi0_default(self) -> torch.Tensor:
        """Default offset for the pressure."""
        return torch.zeros(
            (1, 1, self._space.nx + 1, self._space.ny + 1),
            **defaults.get(),
        )

    @property
    def offset_psi0(self) -> torch.Tensor:
        """Offset for the pressure."""
        try:
            return self._offset_psi0
        except AttributeError:
            return self.offset_psi0_default

    @offset_psi0.setter
    def offset_psi0(self, value: torch.Tensor) -> None:
        """Set the offset for the pressure."""
        warnings.warn(
            f"Setting a value to {self.__class__.__name__}.offset_psi0"
            " is very likely yo cause overflow.",
            stacklevel=1,
        )
        self._offset_psi0 = value

    @cached_property
    def offset_psi1_default(self) -> torch.Tensor:
        """Default offset for the pressure."""
        return torch.zeros(
            (1, 1, self._space.nx + 1, self._space.ny + 1),
            **defaults.get(),
        )

    @property
    def offset_psi1(self) -> torch.Tensor:
        """Offset for the pressure."""
        try:
            return self._offset_psi1
        except AttributeError:
            return self.offset_psi1_default

    @offset_psi1.setter
    def offset_psi1(self, value: torch.Tensor) -> None:
        """Set the offset for the pressure."""
        warnings.warn(
            f"Setting a value to {self.__class__.__name__}.offset_psi1"
            " is very likely yo cause overflow.",
            stacklevel=1,
        )
        self._offset_psi1 = value

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
        self._set_io()
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
        psi_filt = self._filter(psi[0, 0] - self.offset_psi0[0, 0])
        psi_filt = psi_filt.unsqueeze(0).unsqueeze(0)
        psi2 = self.alpha * psi_filt + self.offset_psi1
        source_term = -1 * self.beta_plane.f0**2 * psi2 / self._H1 / self._g2
        beta_effect = self.beta_plane.beta * (self._y - self._y0)
        return self.masks.h * (
            self._points_to_surfaces(
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
            psi_filt = self._filter(psi_i[0, 0] - self.offset_psi0[0, 0])
            psi_filt = psi_filt.unsqueeze(0).unsqueeze(0)
            psi2 = self.alpha * psi_filt + self.offset_psi1
            psi2_interp = self._points_to_surfaces(
                self._points_to_surfaces(psi2),
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
            -self._points_to_surfaces(psi_modes).mean((-1, -2), keepdim=True)
            / self.homsol_mean
        )
        psi_modes += gamma * self.homsol
        return torch.einsum("lm,...mxy->...lxy", self.Cm2l, psi_modes)
