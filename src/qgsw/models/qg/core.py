"""Pytorch multilayer QG as projected SW, Louis Thiry, 9. oct. 2023.

- QG herits from SW class, prognostic variables: u, v, h
- DST spectral solver for QG elliptic equation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from qgsw import verbose
from qgsw.models.core.finite_diff import grad_perp
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.sw import SW

if TYPE_CHECKING:
    from pathlib import Path


class QG(SW):
    """Multilayer quasi-geostrophic model as projected SW.

    Following https://doi.org/10.1029/2021MS002663 .

    Physical Variables are :
        - u_phys: Zonal velocity
        - v_phys: Meridional Velocity
        - h_phys: layers thickness

    Prognostic Variables are linked to physical variables through:
        - u = u_phys x dx
        - v = v_phys x dy
        - h = h_phys x dx x dy

    Diagnostic variables are:
        - U = u_phys / dx
        - V = v_phys / dx
        - omega = omega_phys x dx x dy    (rel. vorticity)
        - eta = eta_phys                  (interface height)
        - p = p_phys                      (hydrostratic pressure)
        - k_energy = k_energy_phys        (kinetic energy)
        - pv = pv_phys                    (potential vorticity)

    References variables are denoted with the subscript _ref:
        - h_ref
        - eta_ref
        - p_ref
        - h_ref_ugrid
        - h_ref_vgrid
        - dx_p_ref
        - dy_p_ref
    """

    def __init__(self, param: dict[str, Any]) -> None:
        """Parameters

        param: python dict. with following keys
            'H':        Tensor (nl,) or (nl, nx, ny),
            unperturbed layer thickness
            'g_prime':  Tensor (nl,), reduced gravities
            'f':        Tensor (nx, ny), Coriolis parameter
            'taux':     float or Tensor (nx-1, ny), top-layer forcing,
            x component
            'tauy':     float or Tensor (nx, ny-1), top-layer forcing,
            y component
            'dt':       float > 0., integration time-step
            'n_ens':    int, number of ensemble member
            'device':   'str', torch devicee e.g. 'cpu', 'cuda', 'cuda:0'
            'dtype':    torch.float32 of torch.float64
            'slip_coef':    float, 1 for free slip, 0 for no-slip,
            inbetween for
                        partial free slip.
        'bottom_drag_coef': float, linear bottom drag coefficient
        """
        super().__init__(param)

        verbose.display(
            msg="class QG, ignoring barotropic filter",
            trigger_level=2,
        )

        # init matrices for elliptic equation
        self.compute_auxillary_matrices()

        # precompile functions
        self.grad_perp = torch.jit.trace(grad_perp, (self.p,))  # ?

    def _validate_layers(self, h: torch.Tensor) -> torch.Tensor:
        """Perform additional validation over H.

        Args:
            h (torch.Tensor): Layers thickness.

        Raises:
            ValueError: if H is not constant in space

        Returns:
            torch.Tensor: H
        """
        h = super()._validate_layers(h)
        if h.shape[-2:] != (1, 1):
            msg = (
                "H must me constant in space for "
                "qg approximation, i.e. have shape (...,1,1)"
                f"got shape shape {h.shape}"
            )
            raise InvalidLayersDefinitionError(msg)
        return h

    def _compute_A(  # noqa: N802
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
        if self.space.nl == 1:
            return torch.tensor([[1.0 / (H * g_prime)]], **self.arr_kwargs)
        A = torch.zeros((self.space.nl, self.space.nl), **self.arr_kwargs)  # noqa: N806
        A[0, 0] = 1.0 / (H[0] * g_prime[0]) + 1.0 / (H[0] * g_prime[1])
        A[0, 1] = -1.0 / (H[0] * g_prime[1])
        for i in range(1, self.space.nl - 1):
            A[i, i - 1] = -1.0 / (H[i] * g_prime[i])
            A[i, i] = 1.0 / H[i] * (1 / g_prime[i + 1] + 1 / g_prime[i])
            A[i, i + 1] = -1.0 / (H[i] * g_prime[i + 1])
        A[-1, -1] = 1.0 / (H[self.space.nl - 1] * g_prime[self.space.nl - 1])
        A[-1, -2] = -1.0 / (H[self.space.nl - 1] * g_prime[self.space.nl - 1])
        return A

    def _compute_layers_to_mode_decomposition_matrices(
        self,
        A: torch.Tensor,  # noqa: N803
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Layers to mode decomposition.

        A = Cm2l @ Λ @ Cl2m

        Args:
            A (torch.Tensor): Stretching Operator

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Cm2l, Λ, Cl2m
        """
        # layer-to-mode and mode-to-layer matrices
        lambd_r, R = torch.linalg.eig(A)  # noqa: N806
        _, L = torch.linalg.eig(A.T)  # noqa: N806
        lambd: torch.Tensor = lambd_r.real.reshape((1, self.space.nl, 1, 1))
        with np.printoptions(precision=1):
            radius = (
                1e-3 / torch.sqrt(self.f0**2 * lambd.squeeze()).cpu().numpy()
            )
            verbose.display(
                msg=f"Rossby deformation Radii (km): {radius}",
                trigger_level=2,
            )
        R, L = R.real, L.real  # noqa: N806
        # Diagonalization of A: A = Cm2l @ Λ @ Cl2m
        # layer to mode: pseudo inverse of R
        Cl2m = torch.diag(1.0 / torch.diag(L.T @ R)) @ L.T  # noqa: N806
        # mode to layer
        Cm2l = R  # noqa: N806
        return Cm2l, lambd, Cl2m

    def compute_auxillary_matrices(self) -> None:
        """More informations on the process here : https://gmd.copernicus.org/articles/17/1749/2024/."""
        # A operator
        self.A = self._compute_A(self.H.squeeze(), self.g_prime.squeeze())

        # layer-to-mode and mode-to-layer matrices
        decomp = self._compute_layers_to_mode_decomposition_matrices(self.A)
        self.Cm2l, self.lambd, self.Cl2m = decomp

        # Governing Equation: ∆Ψ - (f_0)² A Ψ = q - βy
        # With Diagonalization: ∆Ψ - (f_0)² Cm2l @ Λ @ Cl2m Ψ = q - βy
        # Layer to mode transform: Ψ_m = Cl2m @ Ψ ; q_m = Cl2m @ q
        # Within Diagonalized Equation: ∆Ψ_m - (f_0)² Λ @ Ψ_m = q_m - βy
        # In Fourier Space: "(∆ - (f_0)² Λ) @ Ψ_m = q_m - βy"

        # For Helmholtz equations
        nl, nx, ny = self.space.nl, self.space.nx, self.space.ny
        laplace_dstI = (  # noqa: N806
            compute_laplace_dstI(
                nx,
                ny,
                self.space.dx,
                self.space.dy,
                self.arr_kwargs,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Compute "(∆ - (f_0)² Λ)" in Fourier Space
        self.helmholtz_dstI = laplace_dstI - self.f0**2 * self.lambd
        # Constant Omega grid
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
                cst_wgrid[..., 1:-1, 1:-1],
                self.helmholtz_dstI,
            )
        # Compute homogenous solution
        self.homsol_wgrid = cst_wgrid + sol_wgrid * self.f0**2 * self.lambd
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1, -2), keepdim=True)
        self.homsol_hgrid = self.cell_corners_to_cell_centers(
            self.homsol_wgrid,
        )
        self.homsol_hgrid_mean = self.homsol_hgrid.mean((-1, -2), keepdim=True)

    def _add_wind_forcing(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Wind forcing to du and dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with wind forcing
        """
        du[..., 0, :, :] += self.taux / self.H[0] * self.space.dx
        dv[..., 0, :, :] += self.tauy / self.H[0] * self.space.dy
        return du, dv

    def set_physical_uvh(
        self,
        u_phys: torch.Tensor,  # noqa: ARG002
        v_phys: torch.Tensor,  # noqa: ARG002
        h_phys: torch.Tensor,  # noqa: ARG002
    ) -> None:
        """Set the physical u,v and h.

        Args:
            u_phys (torch.Tensor): useless, for compatibilty reasons only.
            v_phys (torch.Tensor): useless, for compatibilty reasons only.
            h_phys (torch.Tensor): useless, for compatibilty reasons only.
        """
        super().compute_time_derivatives()
        self.u, self.v, self.h = self.project_qg(self.u, self.v, self.h)
        self.compute_diagnostic_variables()

    def G(  # noqa: N802
        self,
        p: torch.Tensor,
        p_i: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """G operator.

        Args:
            p (torch.Tensor): Pressure.
            p_i (Union[None, torch.Tensor], optional): Interpolated pressure
             ("middle of grid cell"). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        p_i = self.cell_corners_to_cell_centers(p) if p_i is None else p_i
        dx, dy = self.space.dx, self.space.dy

        # geostrophic balance
        u = -torch.diff(p, dim=-1) / dy / self.f0 * dx
        v = torch.diff(p, dim=-2) / dx / self.f0 * dy
        # h = diag(H)Ap
        h = (
            self.H
            * torch.einsum("lm,...mxy->...lxy", self.A, p_i)
            * self.space.area
        )

        return u, v, h

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
        p_qg_i = self.cell_corners_to_cell_centers(p_qg)
        return p_qg, p_qg_i

    def Q(  # noqa: N802
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
        f0, H, area = self.f0, self.H, self.space.area  # noqa: N806
        # Compute ω = ∂_x v - ∂_y u
        omega = torch.diff(v[..., 1:-1], dim=-2) - torch.diff(
            u[..., 1:-1, :],
            dim=-1,
        )
        # Compute ω-f_0*h/H
        return (omega - f0 * self.cell_corners_to_cell_centers(h) / H) * (
            f0 / area
        )

    def project_qg(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QG projector P = G o (Q o G)^{-1} o Q.

        Args:
            u (torch.Tensor): Zonal velocity
            v (torch.Tensor): Meridional velocity
            h (torch.Tensor): Layers Thickness

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Quasi geostrophic
            u,v and h
        """
        return self.G(*self.QoG_inv(self.Q(u, v, h)))

    def compute_ageostrophic_velocity(
        self,
        dt_uvh_qg: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dt_uvh_sw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Compute ageostrophic variables.

        Computes:
        - Ageostrophic Zonal Velocity u_a
        - Ageostrophic Meridional Velocity v_a
        - Ageostrophic Kinetic Energy k_energy_a
        - Ageostrophic Vorticity omega_a
        - Ageostrophic Divergence div_a

        Args:
            dt_uvh_qg (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): u,v,h
            after qg projection
            dt_uvh_sw (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): u,v,h
            before projection
        """
        self.u_a = -(dt_uvh_qg[1] - dt_uvh_sw[1]) / self.f0 / self.space.dy
        self.v_a = (dt_uvh_qg[0] - dt_uvh_sw[0]) / self.f0 / self.space.dx
        self.k_energy_a = 0.25 * (
            self.u_a[..., 1:] ** 2
            + self.u_a[..., :-1] ** 2
            + self.v_a[..., 1:, :] ** 2
            + self.v_a[..., :-1, :] ** 2
        )
        self.omega_a = (
            torch.diff(self.v_a, dim=-2) / self.space.dx
            - torch.diff(self.u_a, dim=-1) / self.space.dy
        )
        self.div_a = (
            torch.diff(self.u_a[..., 1:-1], dim=-2) / self.space.dx
            + torch.diff(self.v_a[..., 1:-1, :], dim=-1) / self.space.dy
        )

    def compute_pv(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Shallow Water Potential Vorticty.

        Args:
            h (torch.Tensor): Prognostic layer thickness perturbation.

        Returns:
            torch.Tensor: Potential Vorticity
        """
        beta_y = self.cell_corners_to_cell_centers(
            (self.f - self.f0).unsqueeze(0),
        )
        omega = self.cell_corners_to_cell_centers(self.omega)
        return beta_y + omega - self.f0 * h / self.h_ref

    def compute_diagnostic_variables(self) -> None:
        """Compute Diagnostic Variables.

        Compute the model's diagnostic variables.

        Computed variables:
        - Vorticity: omega
        - Interface heights: eta
        - Pressure: p
        - Zonal velocity: U
        - Meridional velocity: V
        - Zonal Velocity Momentum: U_m
        - Meriodional Velocity Momentum: V_m
        - Kinetic Energy: k_energy
        - Potential Vorticity: pv

        Compute the result given the prognostic
        variables self.u, self.v, self.h .

        """
        super().compute_diagnostic_variables()
        self.pv = self.compute_pv(self.h)

    def compute_time_derivatives(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: dt_u, dt_v, dt_h
        """
        dt_uvh_sw = super().compute_time_derivatives()
        dt_uvh_qg = self.project_qg(*dt_uvh_sw)
        self.dt_h = dt_uvh_sw[2]
        self.P_dt_h = dt_uvh_qg[2]
        self.P2_dt_h = self.project_qg(*dt_uvh_qg)[2]

        self.compute_ageostrophic_velocity(dt_uvh_qg, dt_uvh_sw)

        return dt_uvh_qg

    def save_uv_ageostrophic(self, output_file: Path) -> None:
        """Save U ageostrophic and V ageostrophic to a given file.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        np.savez(
            output_file,
            u=self.u_a.cpu().numpy().astype("float32"),
            v=self.v_a.cpu().numpy().astype("float32"),
        )

        verbose.display(msg=f"saved u_a,v_a to {output_file}", trigger_level=1)

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        super().step()
