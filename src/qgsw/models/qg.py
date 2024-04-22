# ruff: noqa
"""
Pytorch multilayer QG as projected SW, Louis Thiry, 9. oct. 2023.
  - QG herits from SW class, prognostic variables: u, v, h
  - DST spectral solver for QG elliptic equation
"""

import numpy as np
import torch
import torch.nn.functional as F

from qgsw.helmholtz import (
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    dstI2D,
    solve_helmholtz_dstI_cmm,
    compute_capacitance_matrices,
)
from qgsw.finite_diff import grad_perp
from qgsw.models.sw import SW
from typing import Any, Union


class QG(SW):
    """Multilayer quasi-geostrophic model as projected SW."""

    def __init__(self, param):
        super().__init__(param)

        # init matrices for elliptic equation
        self.compute_auxillary_matrices()

        # precompile functions
        self.grad_perp = torch.jit.trace(grad_perp, (self.p,))  # ?

    def _validate_H(self, param: dict[str, Any], key: str) -> torch.Tensor:
        """Perform additional validation over H.

        Args:
            param (dict[str, Any]): Parameters dict.
            key (str): Key for H value.

        Raises:
            ValueError: if H is not constant in space

        Returns:
            torch.Tensor: H
        """
        value = super()._validate_H(param, key)
        if value.shape[-2:] != (1, 1):
            msg = (
                "H must me constant in space for "
                "qg approximation, i.e. have shape (...,1,1)"
                f"got shape shape {value.shape}"
            )
            raise ValueError(msg)
        return value

    def compute_auxillary_matrices(self):
        """More informations on the process here : https://gmd.copernicus.org/articles/17/1749/2024/."""
        # A operator
        H, g_prime = self.H.squeeze(), self.g_prime.squeeze()
        self.A = torch.zeros((self.nl, self.nl), **self.arr_kwargs)
        if self.nl == 1:
            self.A[0, 0] = 1.0 / (H * g_prime)
        else:
            self.A[0, 0] = 1.0 / (H[0] * g_prime[0]) + 1.0 / (
                H[0] * g_prime[1]
            )
            self.A[0, 1] = -1.0 / (H[0] * g_prime[1])
            for i in range(1, self.nl - 1):
                self.A[i, i - 1] = -1.0 / (H[i] * g_prime[i])
                self.A[i, i] = (
                    1.0 / H[i] * (1 / g_prime[i + 1] + 1 / g_prime[i])
                )
                self.A[i, i + 1] = -1.0 / (H[i] * g_prime[i + 1])
            self.A[-1, -1] = 1.0 / (H[self.nl - 1] * g_prime[self.nl - 1])
            self.A[-1, -2] = -1.0 / (H[self.nl - 1] * g_prime[self.nl - 1])

            # # equivalent computation without for loop
            # # Invert matrices
            # H_inv = 1 / H
            # g_inv = 1 / g_prime

            # # Compute products
            # hg = H_inv * g_inv

            # # Pad with zeros
            # hg_shift = H_inv * F.pad(g_inv[1:], (0, 1))

            # # Create diagonal matrices
            # diag0 = torch.diag_embed(
            #     input=hg + hg_shift,
            # )
            # diag1 = torch.diag_embed(
            #     input=-hg_shift[:-1],
            #     offset=1,
            # )
            # diag_1 = torch.diag_embed(
            #     input=-hg[1:],
            #     offset=-1,
            # )

            ## Create final matrice
            # self.A = diag0 + diag1 + diag_1

        # layer-to-mode and mode-to-layer matrices
        lambd_r, R = torch.linalg.eig(self.A)
        lambd_l, L = torch.linalg.eig(self.A.T)
        self.lambd: torch.Tensor = lambd_r.real.reshape((1, self.nl, 1, 1))
        with np.printoptions(precision=1):
            print(
                "  - Rossby deformation Radii (km): ",
                1e-3
                / torch.sqrt(self.f0**2 * self.lambd.squeeze()).cpu().numpy(),
            )
        R, L = R.real, L.real
        # layer to mode
        self.Cl2m = torch.diag(1.0 / torch.diag(L.T @ R)) @ L.T
        # mode to layer
        self.Cm2l = R

        # For Helmholtz equations
        nl, nx, ny = self.nl, self.nx, self.ny
        laplace_dstI = (
            compute_laplace_dstI(nx, ny, self.dx, self.dy, self.arr_kwargs)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.helmholtz_dstI = laplace_dstI - self.f0**2 * self.lambd

        cst_wgrid = torch.ones((1, nl, nx + 1, ny + 1), **self.arr_kwargs)
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
                cst_wgrid[..., 1:-1, 1:-1], self.helmholtz_dstI
            )

        self.homsol_wgrid = cst_wgrid + sol_wgrid * self.f0**2 * self.lambd
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1, -2), keepdim=True)
        self.homsol_hgrid = self.interp_TP(self.homsol_wgrid)
        self.homsol_hgrid_mean = self.homsol_hgrid.mean((-1, -2), keepdim=True)

    def add_wind_forcing(self, du, dv):
        du[..., 0, :, :] += self.taux / self.H[0] * self.dx
        dv[..., 0, :, :] += self.tauy / self.H[0] * self.dy
        return du, dv

    def set_physical_uvh(self, u_phys, v_phys, h_phys):
        super().compute_time_derivatives()
        self.u, self.v, self.h = self.project_qg(self.u, self.v, self.h)
        self.compute_diagnostic_variables()

    def G(
        self, p: torch.Tensor, p_i: Union[None, torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """G operator.

        Args:
            p (torch.Tensor): Pressure.
            p_i (Union[None, torch.Tensor], optional): Interpolated pressure
             ("middle of grid cell"). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        p_i = self.interp_TP(p) if p_i is None else p_i
        dx, dy = self.dx, self.dy

        # geostrophic balance
        u = -torch.diff(p, dim=-1) / dy / self.f0 * dx
        v = torch.diff(p, dim=-2) / dx / self.f0 * dy
        # h = diag(H)Ap
        h = self.H * torch.einsum("lm,...mxy->...lxy", self.A, p_i) * self.area

        return u, v, h

    def QoG_inv(
        self, elliptic_rhs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(Q o G)^{-1} operator: solve elliptic equation with mass conservation.

        More informatiosn: https://gmd.copernicus.org/articles/17/1749/2024/.)

        Args:
            elliptic_rhs (torch.Tensor): Elliptic equation right hand side value (ω-f_0*h/H).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Quasi-geostrophique pressure,
            interpolated quasi-geostroophic pressure ("middle of grid cell").
        """
        # transform to modes
        helmholtz_rhs: torch.Tensor = torch.einsum(
            "lm,...mxy->...lxy", self.Cl2m, elliptic_rhs
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
            "lm,...mxy->...lxy", self.Cm2l, p_modes
        )
        p_qg_i = self.interp_TP(p_qg)
        return p_qg, p_qg_i

    def Q(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Q operator: compute elliptic equation r.h.s.

        Args:
            u (torch.Tensor): Zonal speed.
            v (torch.Tensor): Meridional Speed.
            h (torch.Tensor): Layer  thickness.

        Returns:
            torch.Tensor: Elliptic equation right hand side (ω-f_0*h/H).
        """
        f0, H, area = self.f0, self.H, self.area
        # Compute ω = ∂_x v - ∂_y u
        omega = torch.diff(v[..., 1:-1], dim=-2) - torch.diff(
            u[..., 1:-1, :], dim=-1
        )
        # Compute ω-f_0*h/H
        elliptic_rhs = (omega - f0 * self.interp_TP(h) / H) * (f0 / area)
        return elliptic_rhs

    def project_qg(self, u, v, h):
        """QG projector P = G o (Q o G)^{-1} o Q"""
        return self.G(*self.QoG_inv(self.Q(u, v, h)))

    def compute_ageostrophic_velocity(self, dt_uvh_qg, dt_uvh_sw):
        self.u_a = -(dt_uvh_qg[1] - dt_uvh_sw[1]) / self.f0 / self.dy
        self.v_a = (dt_uvh_qg[0] - dt_uvh_sw[0]) / self.f0 / self.dx
        self.k_energy_a = 0.25 * (
            self.u_a[..., 1:] ** 2
            + self.u_a[..., :-1] ** 2
            + self.v_a[..., 1:, :] ** 2
            + self.v_a[..., :-1, :] ** 2
        )
        self.omega_a = (
            torch.diff(self.v_a, dim=-2) / self.dx
            - torch.diff(self.u_a, dim=-1) / self.dy
        )
        self.div_a = (
            torch.diff(self.u_a[..., 1:-1], dim=-2) / self.dx
            + torch.diff(self.v_a[..., 1:-1, :], dim=-1) / self.dy
        )

    def compute_diagnostic_variables(self):
        super().compute_diagnostic_variables()
        self.pv = self.interp_TP(self.omega) / self.area - self.f0 * (
            self.h / self.h_ref
        )

    def compute_time_derivatives(self):
        dt_uvh_sw = super().compute_time_derivatives()
        dt_uvh_qg = self.project_qg(*dt_uvh_sw)
        self.dt_h = dt_uvh_sw[2]
        self.P_dt_h = dt_uvh_qg[2]
        self.P2_dt_h = self.project_qg(*dt_uvh_qg)[2]

        self.compute_ageostrophic_velocity(dt_uvh_qg, dt_uvh_sw)

        return dt_uvh_qg
