"""Shallow Water models with barotropic filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.fields.variables.tuples import UVH
from qgsw.models.core.helmholtz import HelmholtzNeumannSolver
from qgsw.models.core.helmholtz_multigrid import MG_Helmholtz
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.sw.core import SW
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core import grid_conversion as convert
from qgsw.spatial.core.discretization import SpaceDiscretization2D

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


class BaseSWFilterBarotropic(ABC, SW):
    """Shallow Water with Barotropic Filtering."""

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,
    ) -> None:
        """SWFilterBarotropic Model Instantiation.

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
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )

    @ModelParamChecker.dt.setter
    def dt(self, dt: float) -> None:
        """Timestep setter."""
        ModelParamChecker.dt.fset(self, dt)
        self._set_solver()

    @property
    def tau(self) -> float:
        """Tau value."""
        return 2 * self.dt

    @abstractmethod
    def _set_solver(self) -> None: ...

    @abstractmethod
    def filter_barotropic_waves(
        self,
        dt_uvh: UVH,
    ) -> UVH:
        """Filter barotropic waves by computing damping term.

        Args:
            dt_uvh (torch.Tensor): Derivative of prognostic variables
            u,v and h.
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: filtered dt_uvh
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped
        """

    def compute_time_derivatives(
        self,
        uvh: UVH,
    ) -> UVH:
        """Compute time derivatives of prognostic variables.

        Args:
            uvh (UVH): Prognostic variables u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Derivatives of u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        dt_uvh = super().compute_time_derivatives(uvh)
        return self.filter_barotropic_waves(dt_uvh)


class SWFilterBarotropicSpectral(BaseSWFilterBarotropic):
    """Rotating Shallow Water model with spectral barotropic filtering."""

    _type = ModelName.SW_FILTER_SPECTRAL

    def _set_solver(self) -> None:
        """Set Helmoltz Solver for barotropic and spectral."""
        verbose.display(
            msg="Using barotropic filter in spectral approximation.",
            trigger_level=2,
        )
        H_tot = self.H.sum(dim=-3, keepdim=True)  # noqa: N806
        lambd = 1.0 / (self.g * self.dt * self.tau * H_tot)
        self.helm_solver = HelmholtzNeumannSolver(
            self.space.nx,
            self.space.ny,
            self.space.dx,
            self.space.dy,
            lambd[0, 0],
            self.dtype,
            self.device.get(),
            mask=self.masks.h[0, 0],
        )

    def filter_barotropic_waves(
        self,
        dt_uvh: UVH,
    ) -> UVH:
        """Inspired from https://doi.org/10.1029/2000JC900089.

        Args:
            dt_uvh (torch.Tensor): Derivative of prognostic variables
            u,v and h.
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: filtered dt_uvh
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped
        """
        # Sum h on u grid
        h_tot_ugrid = self.h_ref_ugrid + convert.h_to_u(self.h, self.masks.h)
        # Sum h on v grid
        h_tot_vgrid = self.h_ref_vgrid + convert.h_to_v(self.h, self.masks.h)

        u_star = (self.u + self.dt * dt_uvh.u) / self.space.dx
        v_star = (self.v + self.dt * dt_uvh.v) / self.space.dy
        u_bar_star = (u_star * h_tot_ugrid).sum(
            dim=-3,
            keepdim=True,
        ) / h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * h_tot_vgrid).sum(
            dim=-3,
            keepdim=True,
        ) / h_tot_vgrid.sum(dim=-3, keepdim=True)
        rhs = (
            1.0
            / (self.g * self.dt * self.tau)
            * (
                torch.diff(u_bar_star, dim=-2) / self.space.dx
                + torch.diff(v_bar_star, dim=-1) / self.space.dy
            )
        )
        w_surf_imp = self.helm_solver.solve(rhs)  # ∂_t η
        filt_u = (
            F.pad(
                -self.g * self.tau * torch.diff(w_surf_imp, dim=-2),
                (0, 0, 1, 1),
            )
            * self.masks.u
        )
        filt_v = (
            F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-1), (1, 1))
            * self.masks.v
        )

        return UVH(dt_uvh.u + filt_u, dt_uvh.v + filt_v, dt_uvh.h)


class SWFilterBarotropicExact(BaseSWFilterBarotropic):
    """Rotating Shallow Water model with barotropic filtering."""

    _type = ModelName.SW_FILTER_EXACT

    def _set_solver(self) -> None:
        """Set Helmoltz Solver for barotropic and exact form."""
        verbose.display(
            msg="Using barotropic filter in exact form.",
            trigger_level=2,
        )
        # Sum h on u grid
        h_tot_ugrid = self.h_ref_ugrid + convert.h_to_u(self.h, self.masks.h)
        # Sum h on v grid
        h_tot_vgrid = self.h_ref_vgrid + convert.h_to_v(self.h, self.masks.h)

        coef_ugrid = (h_tot_ugrid * self.masks.u)[0, 0]
        coef_vgrid = (h_tot_vgrid * self.masks.v)[0, 0]
        lambd = 1.0 / (self.g * self.dt * self.tau)
        self.helm_solver = MG_Helmholtz(
            self.space.dx,
            self.space.dy,
            self.space.nx,
            self.space.ny,
            coef_ugrid=coef_ugrid,
            coef_vgrid=coef_vgrid,
            lambd=lambd.unsqueeze(0),
            device=self.device.get(),
            dtype=self.dtype,
            mask=self.masks.h[0, 0],
            niter_bottom=20,
            use_compilation=False,
        )

    def filter_barotropic_waves(self, dt_uvh: UVH) -> UVH:
        """Inspired from https://doi.org/10.1029/2000JC900089.

        Args:
            dt_uvh (torch.Tensor): Derivative of prognostic variables
            u,v and h.
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: filtered dt_uvh
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped
        """
        # Sum h on u grid
        h_tot_ugrid = self.h_ref_ugrid + convert.h_to_u(self.h, self.masks.h)
        # Sum h on v grid
        h_tot_vgrid = self.h_ref_vgrid + convert.h_to_v(self.h, self.masks.h)

        # compute RHS
        u_star = (self.u + self.dt * dt_uvh.u) / self.space.dx
        v_star = (self.v + self.dt * dt_uvh.v) / self.space.dy
        u_bar_star = (u_star * h_tot_ugrid).sum(
            dim=-3,
            keepdim=True,
        ) / h_tot_ugrid.sum(dim=-3, keepdim=True)
        v_bar_star = (v_star * h_tot_vgrid).sum(
            dim=-3,
            keepdim=True,
        ) / h_tot_vgrid.sum(dim=-3, keepdim=True)
        rhs = (
            1.0
            / (self.g * self.dt * self.tau)
            * (
                torch.diff(h_tot_ugrid * u_bar_star, dim=-2) / self.space.dx
                + torch.diff(h_tot_vgrid * v_bar_star, dim=-1) / self.space.dy
            )
        )
        coef_ugrid = (h_tot_ugrid * self.masks.u)[0, 0]
        coef_vgrid = (h_tot_vgrid * self.masks.v)[0, 0]
        w_surf_imp = self.helm_solver.solve(rhs, coef_ugrid, coef_vgrid)
        # WIP

        filt_u = (
            F.pad(
                -self.g * self.tau * torch.diff(w_surf_imp, dim=-2),
                (0, 0, 1, 1),
            )
            * self.masks.u
        )
        filt_v = (
            F.pad(-self.g * self.tau * torch.diff(w_surf_imp, dim=-1), (1, 1))
            * self.masks.v
        )

        return UVH(dt_uvh.u + filt_u, dt_uvh.v + filt_v, dt_uvh.h)
