"""Collinear Filtered Projector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.fields.variables.prognostic_tuples import UVH
from qgsw.filters.high_pass import GaussianHighPass2D
from qgsw.models.qg.projected.modified.collinear.stretching_matrix import (
    compute_A_12,
)
from qgsw.models.qg.projected.projectors.collinear import CollinearQGProjector
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.filters.base import _Filter
    from qgsw.masks import Masks
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


class CollinearFilteredQGProjector(CollinearQGProjector):
    """QG Projector."""

    _sigma = 1

    @with_shapes(
        A=(1, 1),
        H=(2, 1, 1),
        g_prime=(2, 1, 1),
    )
    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        space: SpaceDiscretization3D,
        f0: float,
        masks: Masks,
    ) -> None:
        """Instantiate the projector.

        Args:
            A (torch.Tensor): Stretching matrix.
            H (torch.Tensor): Layers reference thickness.
            g_prime (torch.Tensor): Reduced gravity in the layer 2.
            space (SpaceDiscretization3D): 3D space discretization.
            f0 (float): _description_
            masks (Masks): _description_
        """
        super().__init__(
            A=A,
            H=H,
            g_prime=g_prime,
            space=space,
            f0=f0,
            masks=masks,
        )
        self._filter = self.create_filter(self._sigma)

    @property
    def filter(self) -> GaussianHighPass2D:
        """Filter."""
        return self._filter

    @classmethod
    @with_shapes(
        H=(2, 1, 1),
        A=(1, 1),
        g_prime=(2, 1, 1),
    )
    def G(  # noqa: N802
        cls,
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        dx: float,
        dy: float,
        ds: float,
        f0: float,
        alpha: torch.Tensor,
        filt: _Filter,
        points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
        p_i: torch.Tensor | None = None,
    ) -> UVH:
        """Geostrophic operator.

        Args:
            p (torch.float):Pressure.
                └── (n_ens, nl, nx+1, ny+1)-shaped
            A (torch.Tensor): Stretching matrix.
                └── (nl,nl)-shaped.
            H (torch.Tensor): Layers reference thickness.
                └── (2, 1, 1)-shaped.
            g_prime (torch.Tensor): Reduced gravity.
                └── (2, 1, 1)-shaped.
            dx (float): dx.
            dy (float): dy.
            ds (float): ds.
            f0 (float): f0.
            alpha (torch.Tensor): Collinearity coeffciient.
                └── (nx, ny)-shaped.
            filt (_Filter): Filter.
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface function.
            p_i (torch.Tensor | None, optional): Interpolated pressure.
            Defaults to None.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Prognostic variables u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        p_i = points_to_surfaces(p) if p_i is None else p_i

        # geostrophic balance
        u = -torch.diff(p, dim=-1) / dy / f0 * dx
        v = torch.diff(p, dim=-2) / dx / f0 * dy
        # h = diag(H)Ap
        A_12 = compute_A_12(H[:, 0, 0], g_prime[:, 0, 0])  # noqa: N806
        p_i_filt = filt(p_i[0, 0]).unsqueeze(0).unsqueeze(0)
        h = (
            H[0, 0, 0]
            * (
                torch.einsum("lm,...mxy->...lxy", A, p_i)
                + A_12 * alpha * p_i_filt
            )
            * ds
        )

        return UVH(u, v, h)

    def _G(self, p: torch.Tensor, p_i: torch.Tensor | None) -> UVH:  # noqa: N802
        """Geostrophic operator.

        Args:
            p (torch.float):Pressure, (n_ens, nl, nx+1, ny+1)-shaped.
                └── (n_ens, nl, nx+1, ny+1)-shaped
            p_i (torch.Tensor | None): Interpolated pressure.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Prognostic variables u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return self.G(
            p=p,
            p_i=p_i,
            A=self._A,
            H=self._H,
            g_prime=self._g_prime,
            dx=self._space.dx,
            dy=self._space.dy,
            ds=self._space.ds,
            f0=self._f0,
            alpha=self.alpha,
            filt=self.filter,
            points_to_surfaces=self._points_to_surface,
        )

    def QoG_inv(  # noqa: N802
        self,
        elliptic_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inversion of Q∘G.

        Args:
            elliptic_rhs (torch.Tensor): Right hand side,
                └── (n_ens, nl, nx-1, ny-1)-shaped.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pressure p and
            interpolated pressure p_i.
                ├── p: (n_ens, nl, nx+1, ny+1)-shaped
                └── p_i: (n_ens, nl, nx, ny)-shaped
        """
        pi_i = torch.zeros(
            (1, 1, self._space.nx, self._space.ny),
            dtype=elliptic_rhs.dtype,
            device=elliptic_rhs.device,
        )
        A_12 = compute_A_12(self.H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806

        verbose.display(
            f"[{self.__class__.__name__}.QoG_inv]: "
            "Retrieving pressure using iterative solving "
            f"with at most {self._MAX_ITERATIONS} iterations.",
            trigger_level=3,
        )

        for k in range(1, self._MAX_ITERATIONS + 1):
            # Since A.shape = (1,1) -> solving in mode space is the same
            # as solving in physical space
            pi_i_filt = self.filter(pi_i[0, 0]).unsqueeze(0).unsqueeze(0)
            pi1 = self._compute_p_modes(
                elliptic_rhs
                + (
                    self._f0**2
                    * A_12
                    * self._points_to_surface(pi_i_filt)
                    * self._points_to_surface(self.alpha)
                ),
            )

            if torch.isnan(pi1).any():
                msg = f"Overflow after {k} iterations."
                raise OverflowError(msg)

            pi1_i = self._points_to_surface(pi1)

            if torch.isclose(
                pi1_i,
                pi_i,
                atol=self._ATOL,
                rtol=self._RTOL,
            ).all():
                verbose.display(
                    f"[{self.__class__.__name__}.QoG_inv]: "
                    f"Convergence reached after {k + 1} iterations.",
                    trigger_level=3,
                )
                break
            pi_i = pi1_i

            if k == self._MAX_ITERATIONS:
                msg = "Max iterations reached, no convergence."
                raise RuntimeError(msg)

        # Add homogeneous solutions to ensure mass conservation
        gamma = -pi1.mean((-1, -2), keepdim=True) / self.homsol_wgrid_mean
        pi1 += gamma * self.homsol_wgrid

        p_qg_i = self._points_to_surface(pi1)
        return pi1, p_qg_i

    @classmethod
    def create_filter(
        cls,
        sigma: float,
    ) -> GaussianHighPass2D:
        """Create filter.

        Args:
            sigma (float): Filter standard deviation.

        Returns:
            SpectralGaussianHighPass2D: Filter.
        """
        return GaussianHighPass2D(sigma=sigma)
