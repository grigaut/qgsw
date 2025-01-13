"""Quasi Geostrophic Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.uvh import UVH, PrognosticTuple
from qgsw.models.base import Model
from qgsw.models.core import schemes
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_layers_to_mode_decomposition,
)
from qgsw.models.sw.core import SW
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


def G(  # noqa: N802
    p: torch.Tensor,
    space: SpaceDiscretization2D,
    H: torch.Tensor,  # noqa: N803
    A: torch.Tensor,  # noqa: N803
    f0: float,
    p_i: torch.Tensor = None,
    corner_to_center: Callable = points_to_surfaces,
) -> UVH:
    """G operator.

    Args:
        p (torch.Tensor): Pressure.
        space (SpaceDiscretization2D): Space Discretization
        H (torch.Tensor): H tensor, (nl,nx,ny) shaped.
        A (torch.Tensor): Stretching operator matrix, (nl,nl) shaped.
        f0 (float): Coriolis parameter.
        p_i (torch.Tensor, optional): Interpolated pressure
        ("middle of grid cell"). Defaults to None.
        corner_to_center (Callable, optional): Corner to center interpolation
        function. Defaults to points_to_surfaces.

    Returns:
        UVH: Resulting u v and h
    """
    p_i = corner_to_center(p) if p_i is None else p_i
    dx, dy = space.dx, space.dy

    # geostrophic balance
    u = -torch.diff(p, dim=-1) / dy / f0 * dx
    v = torch.diff(p, dim=-2) / dx / f0 * dy
    # h = diag(H)Ap
    h = H * torch.einsum("lm,...mxy->...lxy", A, p_i) * space.ds

    return UVH(u, v, h)


class QG(Model):
    """Quasi Geostrophic Model."""

    _type = "QG"

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """QG Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )
        self.A = self.compute_A(H, g_prime)
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )
        decomposition = compute_layers_to_mode_decomposition(self.A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))

    @property
    def sw(self) -> SW:
        """Core Shallow Water Model."""
        return self._core

    @property
    def lambd(self) -> torch.Tensor:
        """Eigenvalues of A."""
        return self._lambd

    @Model.beta_plane.setter
    def beta_plane(self, beta_plane: BetaPlane) -> None:
        """Beta-plane setter."""
        Model.beta_plane.fset(self, beta_plane)
        self.sw.beta_plane = beta_plane
        self.set_helmholtz_solver(self.lambd)

    @Model.slip_coef.setter
    def slip_coef(self, slip_coef: float) -> None:
        """Slip coefficient setter."""
        Model.slip_coef.fset(self, slip_coef)
        self.sw.slip_coef = slip_coef

    @Model.bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag_coef: float) -> None:
        """Beta-plane setter."""
        Model.bottom_drag_coef.fset(self, bottom_drag_coef)
        self.sw.bottom_drag_coef = bottom_drag_coef

    @Model.dt.setter
    def dt(self, dt: float) -> None:
        """Timesetp setter."""
        Model.dt.fset(self, dt)
        self.sw.dt = dt

    @Model.masks.setter
    def masks(self, masks: torch.Tensor) -> None:
        """Masks setter."""
        Model.masks.fset(self, masks)
        self.sw.masks = masks

    @Model.n_ens.setter
    def n_ens(self, n_ens: int) -> None:
        """Ensemble number setter."""
        Model.n_ens.fset(self, n_ens)
        self.sw.n_ens = n_ens

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        msg_parts = [
            f"Model: {self.__class__}",
            f"├── Data type: {self.dtype}",
            f"├── Device: {self.device}",
            (
                f"├── Beta plane: f0 = {self.beta_plane.f0} "
                f"- β = {self.beta_plane.beta}"
            ),
            f"├── dt: {self.dt} s",
        ]
        space_repr_ = self.space.get_repr_parts()
        space_repr = ["├── " + space_repr_.pop(0)]
        space_repr = space_repr + ["│\t" + txt for txt in space_repr_]
        state_repr_ = self._state.get_repr_parts()
        state_repr = ["├── " + state_repr_.pop(0)]
        state_repr = state_repr + ["│\t" + txt for txt in state_repr_]
        sw_repr_ = self.sw.get_repr_parts()
        sw_repr = ["└── Core " + sw_repr_.pop(0)]
        sw_repr = sw_repr + ["\t" + txt for txt in sw_repr_]

        return msg_parts + space_repr + state_repr + sw_repr

    def compute_A(  # noqa: N802
        self,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the stretching operator matrix A.

        Args:
            H (torch.Tensor): Layers reference height.
            g_prime (torch.Tensor): Reduced gravity values.

        Returns:
            torch.Tensor: Stretching Operator
        """
        return compute_A(
            H=H,
            g_prime=g_prime,
            dtype=self.dtype,
            device=self.device.get(),
        )

    def set_helmholtz_solver(self, lambd: torch.Tensor) -> None:
        """Set the Helmholtz Solver.

        Args:
            lambd (torch.Tensor): Matrix A's eigenvalues.
        """
        # For Helmholtz equations
        nl, nx, ny = lambd.shape[1], self.space.nx, self.space.ny
        laplace_dstI = (  # noqa: N806
            compute_laplace_dstI(
                nx,
                ny,
                self.space.dx,
                self.space.dy,
                dtype=self.dtype,
                device=self.device.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Compute "(∆ - (f_0)² Λ)" in Fourier Space
        self.helmholtz_dstI = laplace_dstI - self.beta_plane.f0**2 * lambd
        # Constant Omega grid
        cst_wgrid = torch.ones(
            (1, nl, nx + 1, ny + 1),
            dtype=self.dtype,
            device=self.device.get(),
        )
        if len(self.masks.psi_irrbound_xids) > 0:
            # Handle Non rectangular geometry
            self.cap_matrices = compute_capacitance_matrices(
                self.helmholtz_dstI,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
            )
            sol_wgrid = solve_helmholtz_dstI_cmm(
                (cst_wgrid * self.masks.psi)[..., 1:-1, 1:-1],
                self.helmholtz_dstI,
                self.cap_matrices,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
                self.masks.psi,
            )
        else:
            self.cap_matrices = None
            sol_wgrid = solve_helmholtz_dstI(
                cst_wgrid[..., 1:-1, 1:-1],
                self.helmholtz_dstI,
            )
        # Compute homogenous solution
        self.homsol_wgrid = (
            cst_wgrid + sol_wgrid * self.beta_plane.f0**2 * lambd
        )
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1, -2), keepdim=True)
        self.homsol_hgrid = self.points_to_surfaces(
            self.homsol_wgrid,
        )
        self.homsol_hgrid_mean = self.homsol_hgrid.mean((-1, -2), keepdim=True)

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
            u (torch.Tensor): State variable u.
            v (torch.Tensor): State variable v.
            h (torch.Tensor): State variable h.
        """
        self.sw.set_uvh(u, v, h)
        super().set_uvh(u, v, h)

    def G(  # noqa: N802
        self,
        p: torch.Tensor,
        p_i: torch.Tensor | None = None,
    ) -> UVH:
        """G operator.

        Args:
            p (torch.Tensor): Pressure.
            p_i (Union[None, torch.Tensor], optional): Interpolated pressure
             ("middle of grid cell"). Defaults to None.

        Returns:
            UVH: u, v and h
        """
        return G(
            p,
            self.space,
            self.H,
            self.A,
            self.beta_plane.f0,
            p_i,
            self.points_to_surfaces,
        )

    def QoG_inv(  # noqa: N802
        self,
        elliptic_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(Q o G)^{-1} operator: solve elliptic eq with mass conservation.

        More informatiosn: https://gmd.copernicus.org/articles/17/1749/2024/.)

        Args:
            elliptic_rhs (torch.Tensor): Elliptic equation right hand side
            value (ω-f_0*h/H).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Quasi-geostrophique pressure,
            interpolated quasi-geostroophic pressure ("middle of grid cell").
        """
        # transform to modes
        helmholtz_rhs: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cl2m,
            elliptic_rhs,
        )
        if self.cap_matrices is not None:
            p_modes = solve_helmholtz_dstI_cmm(
                helmholtz_rhs * self.masks.psi[..., 1:-1, 1:-1],
                self.helmholtz_dstI,
                self.cap_matrices,
                self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids,
                self.masks.psi,
            )
        else:
            p_modes = solve_helmholtz_dstI(helmholtz_rhs, self.helmholtz_dstI)

        # Add homogeneous solutions to ensure mass conservation
        alpha = -p_modes.mean((-1, -2), keepdim=True) / self.homsol_wgrid_mean
        p_modes += alpha * self.homsol_wgrid
        # transform back to layers
        p_qg: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy",
            self.Cm2l,
            p_modes,
        )
        p_qg_i = self.points_to_surfaces(p_qg)
        return p_qg, p_qg_i

    def Q(  # noqa: N802
        self,
        uvh: UVH,
    ) -> torch.Tensor:
        """Q operator: compute elliptic equation r.h.s.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            torch.Tensor: Elliptic equation right hand side (ω-f_0*h/H).
        """
        f0, H, ds = self.beta_plane.f0, self.H, self.space.ds  # noqa: N806
        # Compute ω = ∂_x v - ∂_y u
        omega = torch.diff(uvh.v[..., 1:-1], dim=-2) - torch.diff(
            uvh.u[..., 1:-1, :],
            dim=-1,
        )
        # Compute ω-f_0*h/H
        return (omega - f0 * self.points_to_surfaces(uvh.h) / H) * (f0 / ds)

    def project(
        self,
        uvh: UVH,
    ) -> UVH:
        """QG projector P = G o (Q o G)^{-1} o Q.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: Quasi geostrophic u,v and h
        """
        return self.G(*self.QoG_inv(self.Q(uvh)))

    def _init_core_model(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool,  # noqa: FBT001
    ) -> SW:
        """Initialize the core Shallow Water model.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.

        Returns:
            SW: Core model.
        """
        return SW(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )

    def compute_time_derivatives(
        self,
        prognostic: PrognosticTuple,
    ) -> UVH:
        """Compute the prognostic variables derivatives dt_u, dt_v, dt_h.

        Args:
            prognostic (PrognosticTuple): u,v and h.

        Returns:
            UVH: dt_u, dt_v, dt_h
        """
        dt_prognostic_sw = self.sw.compute_time_derivatives(prognostic)
        return self.project(dt_prognostic_sw)

    def update(self, uvh: UVH) -> UVH:
        """Update uvh.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: update prognostic variables.
        """
        return schemes.rk3_ssp(
            uvh,
            self.dt,
            self.compute_time_derivatives,
        )

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
