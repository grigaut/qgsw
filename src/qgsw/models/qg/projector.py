"""QG Projector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.uvh import UVH
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.masks import Masks
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


class QGProjector:
    """QG Projector."""

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        space: SpaceDiscretization3D,
        f0: float,
        masks: Masks,
    ) -> None:
        """Instantiate the projector.

        Args:
            A (torch.Tensor): Stretching matrix.
                └── (nl, nl)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (n_ens, nl, 1, 1)-shaped
            space (SpaceDiscretization3D): 3D space discretization.
            f0 (float): f0.
            masks (Masks): Masks.
        """
        self._H = H
        self._masks = masks
        self._f0 = f0
        self._space = space
        self._points_to_surface = OptimizableFunction[
            torch.Tensor,
            torch.Tensor,
        ](points_to_surfaces)

        self.A = A

    @property
    def A(self) -> torch.Tensor:  # noqa: N802
        """Streching matrix.

        └── (nl,nl)-shaped.
        """
        return self._A

    @A.setter
    def A(self, A: torch.Tensor) -> None:  # noqa: N802, N803
        self._A = A
        decomposition = compute_layers_to_mode_decomposition(A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
        self._set_helmholtz_solver(self.lambd, self._f0)

    @property
    def masks(self) -> Masks:
        """Masks."""
        return self._masks

    @masks.setter
    def masks(self, masks: Masks) -> None:
        self._masks = masks
        self._set_helmholtz_solver(self.lambd, self._f0)

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers reference thickness.

        └── (n_ens, nl, 1, 1)-shaped.
        """
        return self._H

    @property
    def lambd(self) -> torch.Tensor:
        """Eigen values of A.

        └── (1, nl, 1, 1)-shaped.
        """
        return self._lambd

    def _set_helmholtz_solver(self, lambd: torch.Tensor, f0: float) -> None:
        """Set the Helmholtz Solver.

        Args:
            lambd (torch.Tensor): Matrix A's eigenvalues.
                └── (1, nl, 1, 1)-shaped.
            f0 (float): f0.
        """
        # For Helmholtz equations
        nl, nx, ny = self._space.nl, self._space.nx, self._space.ny
        dx, dy = self._space.dx, self._space.dy
        laplace_dstI = (  # noqa: N806
            compute_laplace_dstI(
                nx,
                ny,
                dx,
                dy,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Compute "(∆ - (f_0)² Λ)" in Fourier Space
        helmholtz_dstI = laplace_dstI - f0**2 * lambd  # noqa: N806
        # Constant Omega grid
        cst_wgrid = torch.ones(
            (1, nl, nx + 1, ny + 1),
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        if len(self._masks.psi_irrbound_xids) > 0:
            # Handle Non rectangular geometry
            cap_matrices = compute_capacitance_matrices(
                helmholtz_dstI,
                self._masks.psi_irrbound_xids,
                self._masks.psi_irrbound_yids,
            )
            sol_wgrid = solve_helmholtz_dstI_cmm(
                (cst_wgrid * self._masks.psi)[..., 1:-1, 1:-1],
                helmholtz_dstI,
                cap_matrices,
                self._masks.psi_irrbound_xids,
                self._masks.psi_irrbound_yids,
                self._masks.psi,
            )

            def compute_p_modes(helmholtz_rhs: torch.Tensor) -> torch.Tensor:
                return solve_helmholtz_dstI_cmm(
                    helmholtz_rhs * self._masks.psi[..., 1:-1, 1:-1],
                    helmholtz_dstI,
                    cap_matrices,
                    self._masks.psi_irrbound_xids,
                    self._masks.psi_irrbound_yids,
                    self._masks.psi,
                )
        else:
            sol_wgrid = solve_helmholtz_dstI(
                cst_wgrid[..., 1:-1, 1:-1],
                helmholtz_dstI,
            )

            def compute_p_modes(helmholtz_rhs: torch.Tensor) -> torch.Tensor:
                return solve_helmholtz_dstI(helmholtz_rhs, helmholtz_dstI)

        self._compute_p_modes = compute_p_modes
        # Compute homogenous solution
        self.homsol_wgrid = cst_wgrid + sol_wgrid * f0**2 * lambd
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1, -2), keepdim=True)

    @staticmethod
    def G(  # noqa: N802
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
        ds: float,
        f0: float,
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
        h = H * torch.einsum("lm,...mxy->...lxy", A, p_i) * ds

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
            points_to_surfaces=self._points_to_surface,
        )

    @staticmethod
    def Q(  # noqa: N802
        uvh: UVH,
        H: torch.Tensor,  # noqa: N803
        f0: float,
        ds: float,
        points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (n_ens,nl,1,1)-shaped.
            f0 (float): f0.
            ds (float): ds.
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface interpolation function.

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens,nl,nx,ny)-shaped.
        """
        # Compute ω = ∂_x v - ∂_y u
        omega = torch.diff(uvh.v[..., 1:-1], dim=-2) - torch.diff(
            uvh.u[..., 1:-1, :],
            dim=-1,
        )
        # Compute ω-f_0*h/H
        return (omega - f0 * points_to_surfaces(uvh.h) / H) * (f0 / ds)

    def _Q(self, uvh: UVH) -> torch.Tensor:  # noqa: N802
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens,nl,nx,ny)-shaped.
        """
        return self.Q(
            uvh=uvh,
            f0=self._f0,
            H=self.H,
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

    def project(self, uvh: UVH) -> UVH:
        """Perform projection.

        P(uvh) = (G∘(Q∘G)⁻¹∘Q)(uvh)

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Projected prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return self._G(*self.QoG_inv(self._Q(uvh)))
