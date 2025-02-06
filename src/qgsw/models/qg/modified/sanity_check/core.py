"""Sanity Check model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    Pressure,
    StreamFunction,
)
from qgsw.fields.variables.state import StateAlpha
from qgsw.fields.variables.uvh import UVH, UVHTAlpha
from qgsw.models import matching
from qgsw.models.io import IO
from qgsw.models.qg.core import QG, QGCore
from qgsw.models.qg.modified.exceptions import UnsetAlphaError
from qgsw.models.qg.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.projectors.core import QGProjector
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.masks import Masks
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
        SpaceDiscretization3D,
    )


class QGSanityCheck(QGCore[UVHTAlpha, "QGSanityCheckProjector"]):
    """QG Sanity-Checks."""

    @with_shapes(
        H=(2,),
        g_prime=(2,),
    )
    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: Tensor,  # noqa: N803
        g_prime: Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """TODO.

        Args:
            space_2d (SpaceDiscretization2D): _description_
            H (Tensor): _description_
            g_prime (Tensor): _description_
            beta_plane (BetaPlane): _description_
            FBT002 (_type_): _description_
            optimize (bool, optional): _description_. Defaults to True.
        """
        self._baseline = QG(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )
        QG.__init__(
            self,
            space_2d,
            H[:1],
            compute_g_tilde(g_prime),
            beta_plane,
            optimize,
        )

    @QGCore.dt.setter
    def dt(self, dt: float) -> None:
        """Timestep."""
        QGCore.dt.fset(self, dt)
        self._baseline.dt = dt

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)
        self._P.alpha = alpha

    def set_uvh(self, u: Tensor, v: Tensor, h: Tensor) -> None:
        """TODO.

        Args:
            u (Tensor): _description_
            v (Tensor): _description_
            h (Tensor): _description_

        Returns:
            _type_: _description_
        """
        self._baseline.set_uvh(u, v, h)
        uvh = matching.match_psi(
            UVH(u, v, h),
            self._baseline.g_prime[:, 0, 0],
            compute_g_tilde(self._baseline.g_prime[:, 0, 0]),
        )
        return super().set_uvh(*uvh)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StateAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self._io = IO(
            t=self._state.t,
            u=self._state.u,
            v=self._state.v,
            h=self._state.h,
            alpha=self._state.alpha,
        )

    def _set_projector(self) -> None:
        self._P = QGSanityCheckProjector(
            baseline=self._baseline,
            A=self._baseline.A[:1, :1],
            H=self._baseline.H,
            g_prime=self._baseline.g_prime,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._baseline.step()
        super().step()

    def compute_time_derivatives(self, prognostic: UVH) -> UVH:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Args:
            prognostic (UVH): u,v and h
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: dt_u, dt_v, dt_h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        dt_prognostic_sw = self.sw.compute_time_derivatives(prognostic)
        self.P.uvh_baseline = self._baseline.uvh_dt
        return self._P.project(dt_prognostic_sw)


class QGSanityCheckProjector(QGProjector):
    """QG Sanity-check Projector."""

    def __init__(
        self,
        baseline: QG,
        A: Tensor,  # noqa: N803
        H: Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        space: SpaceDiscretization3D,
        f0: float,
        masks: Masks,
    ) -> None:
        """TODO.

        Args:
            baseline (QG): _description_
            A (Tensor): _description_
            H (Tensor): _description_
            g_prime (Tensor): _description_
            space (SpaceDiscretization3D): _description_
            f0 (float): _description_
            masks (Masks): _description_
        """
        self._baseline = baseline
        self.uvh_baseline = None
        self._g_prime = g_prime
        self._sf_var = StreamFunction(
            Pressure(
                self._baseline.g_prime.unsqueeze(0),
                PhysicalSurfaceHeightAnomaly(
                    PhysicalLayerDepthAnomaly(
                        baseline.space.ds,
                    ),
                ),
            ),
            self._baseline.beta_plane.f0,
        )
        super().__init__(A, H, space, f0, masks)

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        try:
            return self._alpha
        except AttributeError as e:
            raise UnsetAlphaError from e

    @alpha.setter
    @with_shapes(alpha=(1,))
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha

    @classmethod
    def Q(  # noqa: N802
        cls,
        uvh: UVH,
        source_term: torch.Tensor,
        H: Tensor,  # noqa: N803
        f0: float,
        ds: float,
        points_to_surfaces: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """TODO.

        Args:
            uvh (UVH): _description_
            source_term (torch.Tensor): _description_
            H (Tensor): _description_
            f0 (float): _description_
            ds (float): _description_
            points_to_surfaces (Callable[[Tensor], Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        return (
            super().Q(
                uvh,
                H[:1],
                f0,
                ds,
                points_to_surfaces,
            )
            + f0 * source_term
        )

    def _Q(self, uvh: UVH) -> torch.Tensor:  # noqa: N802
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical Potential Vorticity * f0.
                └── (n_ens, nl, nx-1, ny-1)-shaped.
        """
        psi_2_baseline = self._sf_var.compute_no_slice(
            self.uvh_baseline,
        )[:, 1:2, ...]
        self._source_term = (
            self.alpha
            * self._f0**2
            * psi_2_baseline
            / self.H[:1]
            / self._g_prime[1:]
        )
        return self.Q(
            uvh=uvh,
            source_term=self._points_to_surface(self._source_term),
            H=self.H,
            f0=self._f0,
            ds=self._space.ds,
            points_to_surfaces=self._points_to_surface,
        )

    def QoG_inv(  # noqa: N802
        self,
        elliptic_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inversion of Q∘G.

        Args:
            elliptic_rhs (torch.Tensor): Right hand side,
                └── (n_ens, nl, nx, ny)-shaped.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pressure p and
            interpolated pressure p_i.
                ├── p: (n_ens, nl, nx+1, ny+1)-shaped
                └── p_i: (n_ens, nl, nx, ny)-shaped
        """
        # transform to modes
        helmholtz_rhs: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cl2m,
            elliptic_rhs,
        )
        p_modes = self._compute_p_modes(helmholtz_rhs)

        # Add homogeneous solutions to ensure mass conservation
        alpha = -p_modes.mean((-1, -2), keepdim=True) / self.homsol_wgrid_mean
        p_modes += alpha * self.homsol_wgrid
        # transform back to layers
        p_qg: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cm2l,
            p_modes,
        )
        p_qg_i = self._points_to_surface(p_qg)
        return p_qg, p_qg_i

    def _G(self, p: Tensor, p_i: Tensor | None) -> UVH:  # noqa: N802
        return self.G(
            p=p,
            p_i=p_i,
            A=self._A,
            H=self.H[:1],
            dx=self._space.dx,
            dy=self._space.dy,
            ds=self._space.ds,
            f0=self._f0,
            source_term=self._source_term,
            points_to_surfaces=self._points_to_surface,
        )

    @classmethod
    def G(  # noqa: D417, N802
        cls,
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
        ds: float,
        f0: float,
        source_term: torch.Tensor,
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
                └── (n_ens, nl, 1, 1)-shaped.
            dx (float): dx.
            dy (float): dy.
            ds (float): ds.
            f0 (float): f0.
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
        h = H * torch.einsum("lm,...mxy->...lxy", A, p_i - source_term) * ds

        return UVH(u, v, h)
