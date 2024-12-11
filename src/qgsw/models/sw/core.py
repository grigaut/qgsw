"""Shallow-water implementation.

Louis Thiry, Nov 2023 for IFREMER.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.models.base import Model
from qgsw.models.core import finite_diff, schemes
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core import grid_conversion as convert
from qgsw.spatial.core.discretization import SpaceDiscretization3D
from qgsw.variables.uvh import UVH

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


def inv_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of reverse cumsum function.

    Args:
        x (torch.Tensor): Tensor to perform inverse reverse cumsum on.
        dim (int): Dimension to perform the inverse reverse cumsum on.

    Returns:
        torch.Tensor: Inverse Reverse Cumsum
    """
    return torch.cat([-torch.diff(x, dim=dim), x.narrow(dim, -1, 1)], dim=dim)


class SW(Model):
    """# Implementation of multilayer rotating shallow-water model.

    Following https://doi.org/10.1029/2021MS002663 .

    ## Main ingredients
        - vector invariant formulation
        - velocity RHS using vortex force upwinding with wenoz-5 reconstruction
        - mass continuity RHS with finite volume using wenoz-5 recontruction

    ## Variables
    Prognostic variables u, v, h differ from physical variables
    u_phys, v_phys (velocity components) and
    h_phys (layer thickness perturbation) as they include
    metric terms dx and dy :
      - u = u_phys x dx
      - v = v_phys x dy
      - h = g_phys x dx x dy

    Diagnostic variables are :
      - U = u_phys / dx
      - V = v_phys / dx
      - omega = omega_phys x dx x dy    (rel. vorticity)
      - eta = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """

    def __init__(
        self,
        *,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,
    ) -> None:
        """SW Model Instantiation.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            n_ens (int, optional): Number of ensembles. Defaults to 1.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_3d=space_3d,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )

    @property
    def pv(self) -> torch.Tensor:
        """Potential Vorticity."""
        msg = "Shallow Water Models don't have vorticity plots."
        raise AttributeError(msg)

    def _compute_coriolis(self) -> None:
        """Set Coriolis Related Grids."""
        super()._compute_coriolis()
        ## Coriolis grids
        self.f_ugrid = convert.omega_to_u(self.f)
        self.f_vgrid = convert.omega_to_v(self.f)
        self.f_hgrid = convert.omega_to_h(self.f)
        self.fstar_ugrid = self.f_ugrid * self.space.area
        self.fstar_vgrid = self.f_vgrid * self.space.area
        self.fstar_hgrid = self.f_hgrid * self.space.area

    def _add_wind_forcing(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add wind forcing to the derivatives du, dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with wind forcing.
        """
        # Sum h on u grid
        h_tot_ugrid = self.h_ref_ugrid + convert.h_to_u(self.h, self.masks.h)
        # Sum h on v grid
        h_tot_vgrid = self.h_ref_vgrid + convert.h_to_v(self.h, self.masks.h)
        h_ugrid = h_tot_ugrid / self.space.area
        h_vgrid = h_tot_vgrid / self.space.area
        du_wind = self.taux / h_ugrid[..., 0, 1:-1, :] * self.space.dx
        dv_wind = self.tauy / h_vgrid[..., 0, :, 1:-1] * self.space.dy
        du[..., 0, :, :] += du_wind
        dv[..., 0, :, :] += dv_wind
        return du, dv

    def _add_bottom_drag(
        self,
        uvh: UVH,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add bottom drag to the derivatives du, dv.

        Args:
            uvh (UVH): u,v and h.
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with botoom drag forcing.
        """
        du[..., -1, :, :] += -self.bottom_drag_coef * uvh.u[..., -1, 1:-1, :]
        dv[..., -1, :, :] += -self.bottom_drag_coef * uvh.v[..., -1, :, 1:-1]
        return du, dv

    def update(self, uvh: UVH) -> UVH:
        """Performs one step time-integration with RK3-SSP scheme.

        Agrs:
            uvh (UVH): u,v and h.
        """
        return schemes.rk3_ssp(uvh, self.dt, self.compute_time_derivatives)

    def advection_momentum(
        self,
        uvh: UVH,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Use the shallow water momentum equation in rotation frame:
        ∂_t [u v] + (ω + f)x[u v] = -∇(gη + 0.5[u v]⋅[u v])
        Therefore:
        - ∂_t u = (ω + f)v - ∂_x(gη + 0.5(u² + v²))
        - ∂_t v = - (ω + f)u - ∂_y(gη + 0.5(u² + v²))

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection (∂_t u, ∂_t v).
        """
        # Zonal velocity -> corresponds to the v grid
        # Has no value on the boundary of the v grid
        U_m = self.points_to_surfaces(self.U)  # noqa: N806
        # Meridional velocity -> corresponds to the u grid
        # Has no value on the boundary of the u grid
        V_m = self.points_to_surfaces(self.V)  # noqa: N806

        # Vortex-force + Coriolis
        omega_v_m = self._fluxes.w_y(self.omega[..., 1:-1, :], V_m)
        omega_u_m = self._fluxes.w_x(self.omega[..., 1:-1], U_m)

        dt_u = omega_v_m + self.fstar_ugrid[..., 1:-1, :] * V_m
        dt_v = -(omega_u_m + self.fstar_vgrid[..., 1:-1] * U_m)

        # grad pressure + k_energy
        ke_pressure = self.k_energy + self.p
        dt_u -= torch.diff(ke_pressure, dim=-2) + self.dx_p_ref
        dt_v -= torch.diff(ke_pressure, dim=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self._add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self._add_bottom_drag(uvh, dt_u, dt_v)

        return (
            F.pad(dt_u, (0, 0, 1, 1)) * self.masks.u,
            F.pad(dt_v, (1, 1, 0, 0)) * self.masks.v,
        )

    def advection_h(self, h: torch.Tensor) -> torch.Tensor:
        """Advection RHS for thickness perturbation h.

        ∂_t h = - ∇⋅(h_tot u)

        u = [U V]
        h_tot = h_ref + h

        Args:
            h (torch.Tensor): layer Thickness perturbation

        Returns:
            torch.Tensor: h advection (∂_t h).
        """
        h_tot = self.h_ref + h
        # Compute (h_tot x V)
        h_tot_flux_y = self._fluxes.h_y(h_tot, self.V[..., 1:-1])
        # Compute (h_tot x U)
        h_tot_flux_x = self._fluxes.h_x(h_tot, self.U[..., 1:-1, :])
        # Compute -∇⋅(h_tot u) = ∂_x (h_tot x U) + ∂_y (h_tot x V)
        div_no_flux = -finite_diff.div_nofluxbc(h_tot_flux_x, h_tot_flux_y)
        # Apply h mask
        return div_no_flux * self.masks.h

    def compute_time_derivatives(
        self,
        uvh: UVH,
    ) -> UVH:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: dt_u, dt_v, dt_h
        """
        self._state.uvh = uvh
        dt_h = self.advection_h(uvh.h)
        dt_u, dt_v = self.advection_momentum(uvh)
        return UVH(
            dt_u,
            dt_v,
            dt_h,
        )
