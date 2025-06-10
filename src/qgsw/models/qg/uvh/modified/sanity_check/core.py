"""Sanity Check model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.state import StateUVHAlpha
from qgsw.fields.variables.tuples import UVH, UVHTAlpha
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.uvh.core import QG, QGCore
from qgsw.models.qg.uvh.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.masks import Masks
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
        SpaceDiscretization3D,
    )


class QGSanityCheck(
    QGCore[UVHTAlpha, StateUVHAlpha, "QGSanityCheckProjector"],
):
    """QG Sanity-Checks."""

    _type = ModelName.QG_SANITY_CHECK

    @with_shapes(
        H=(2,),
        g_prime=(2,),
    )
    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """QGSanityCheck Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (2,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor.
                └── (2,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        self._baseline = QG(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )
        self._baseline.save_intermediate_p()
        QG.__init__(
            self,
            space_2d,
            H[:1],
            compute_g_tilde(g_prime),
            beta_plane,
            optimize,
        )

    @property
    def baseline(self) -> QG:
        """Baseline model."""
        return self._baseline

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

    @ModelParamChecker.slip_coef.setter
    def slip_coef(self, slip_coefficient: float) -> None:
        """Slip coefficient."""
        ModelParamChecker.slip_coef.fset(self, slip_coefficient)
        self._baseline.slip_coef = slip_coefficient

    @ModelParamChecker.bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag: float) -> None:
        """Bottom drag coefficient."""
        ModelParamChecker.bottom_drag_coef.fset(self, bottom_drag)
        self._baseline.bottom_drag_coef = bottom_drag

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped
        """
        uvh = self._baseline.P.G(
            p[:, : self.space.nl],
            self._baseline.A,
            self._baseline.H,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.points_to_surfaces,
        )
        self.set_uvh(*uvh)

    def set_uvh(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Set u,v,h value from prognostic variables.

        Warning: the expected values are not physical values but prognostic
        values. The variables correspond to the actual self.u, self.v, self.h
        of the model.

        Args:
            u (torch.Tensor): StateUVH variable u.
                └── (n_ens, 2, nx+1, ny)-shaped
            v (torch.Tensor): StateUVH variable v.
                └── (n_ens, 2, nx, ny+1)-shaped
            h (torch.Tensor): StateUVH variable h.
                └── (n_ens, 2, nx, ny)-shaped
        """
        self._baseline.set_uvh(u, v, h)
        return super().set_uvh(u[:, :1], v[:, :1], h[:, :1])

    def _set_io(self, state: StateUVHAlpha) -> None:
        self._io = IO(
            state.t,
            state.alpha,
            *state.physical,
        )

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StateUVHAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )

    def _set_projector(self) -> None:
        """Set the Projector."""
        self._P = QGSanityCheckProjector(
            A=self._baseline.A[:1, :1],
            H=self._baseline.H[:1],
            g_prime=self._baseline.g_prime,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._baseline.step()
        super().step()

    def update(self, prognostic: UVH) -> UVH:
        """Update prognostic.

        Args:
            prognostic (UVH): u,v and h.
                ├── u: (n_ens, 1, nx+1, ny)-shaped
                ├── v: (n_ens, 1, nx, ny+1)-shaped
                └── h: (n_ens, 1, nx, ny)-shaped

        Returns:
            UVH: update prognostic variables.
                ├── u: (n_ens, 1, nx+1, ny)-shaped
                ├── v: (n_ens, 1, nx, ny+1)-shaped
                └── h: (n_ens, 1, nx, ny)-shaped
        """
        self._rk3_i = 0
        return super().update(prognostic)

    def set_wind_forcing(
        self,
        taux: float | torch.Tensor,
        tauy: float | torch.Tensor,
    ) -> None:
        """Set the wind forcing attributes taux and tauy.

        Args:
            taux (float | torch.Tensor): Taux value.
            tauy (float | torch.Tensor): Tauy value.
        """
        super().set_wind_forcing(taux, tauy)
        self.sw.set_wind_forcing(taux, tauy)
        self._baseline.set_wind_forcing(taux, tauy)

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

        p_vals = self._baseline.intermediate_p_values[self._rk3_i]

        self.P.p2_baseline = torch.clone(p_vals[0][:, 1:2])
        self.P.p2_i_baseline = torch.clone(p_vals[1][:, 1:2])
        self._rk3_i += 1
        return self._P.project(dt_prognostic_sw)


class QGSanityCheckProjector(QGProjector):
    """QGProjector for the QGSanityCheck model."""

    p2_baseline: torch.Tensor
    p2_i_baseline: torch.Tensor

    @with_shapes(
        A=(1, 1),
        H=(1, 1, 1),
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
                └── (1, 1)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (1, 1, 1)-shaped
            g_prime (torch.Tensor): Reduced gravity for the 2 layers model.
                └── (2, 1, 1)-shaped
            space (SpaceDiscretization3D): 3D space discretization.
            f0 (float): f0.
            masks (Masks): Masks.
        """
        self._g2 = g_prime[1:2, 0, 0]
        super().__init__(A, H, space, f0, masks)

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
        p2 = self.p2_baseline[..., 1:-1, 1:-1]
        H_1 = self._H[0, 0, 0]  # noqa: N806
        # source_term = f_0^2 / H_1 / g'_2 * p_2
        source_term = self._f0**2 / H_1 / self._g2 * p2
        helmholtz_rhs: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cl2m,
            elliptic_rhs - source_term,
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
        self.p = p_qg
        self.p_i = p_qg_i
        return p_qg, p_qg_i

    @classmethod
    @with_shapes(
        g2=(1,),
        H=(1, 1, 1),
    )
    def G(  # noqa: N802
        cls,
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
        ds: float,
        f0: float,
        g2: torch.Tensor,
        p2_i: torch.Tensor,
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
                └── (nl, 1, 1)-shaped.
            dx (float): dx.
            dy (float): dy.
            ds (float): ds.
            f0 (float): f0.
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface function.
            g2 (torch.Tensor): Reduced gravity in the second layer.
                └── (1,)-shaped
            p2_i (torch.Tensor): Interpolated pressure of second layer.
                └── (n_ens, nl, nx, ny)-shaped
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
        # source_term = A_{1,2} p_2
        source_term = 1 / H[0, 0, 0] / g2 * p2_i
        # h = diag(H)(Ap-A_{1,2}p_2)
        h = H * (torch.einsum("lm,...mxy->...lxy", A, p_i) - source_term) * ds

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
            dx=self._space.dx,
            dy=self._space.dy,
            ds=self._space.ds,
            f0=self._f0,
            p2_i=self.p2_i_baseline,
            g2=self._g2,
            points_to_surfaces=self._points_to_surface,
        )
