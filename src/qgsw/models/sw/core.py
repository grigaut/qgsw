"""Shallow-water implementation.

Louis Thiry, Nov 2023 for IFREMER.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.dynamics import (
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    Pressure,
    SurfaceHeightAnomaly,
    Vorticity,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.energetics import KineticEnergy
from qgsw.fields.variables.uvh import UVH, PrognosticTuple
from qgsw.models.base import Model
from qgsw.models.core import finite_diff, schemes
from qgsw.spatial.core import grid_conversion as convert
from qgsw.spatial.core.discretization import SpaceDiscretization2D

if TYPE_CHECKING:
    from qgsw.fields.variables.state import State
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


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

    _type = "SW"

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,
    ) -> None:
        """SW Model Instantiation.

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
        self.fstar_ugrid = self.f_ugrid * self.space.ds
        self.fstar_vgrid = self.f_vgrid * self.space.ds
        self.fstar_hgrid = self.f_hgrid * self.space.ds

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
        h_ugrid = h_tot_ugrid / self.space.ds
        h_vgrid = h_tot_vgrid / self.space.ds
        du_wind = self.taux / h_ugrid[..., 0, 1:-1, :] * self.space.dx
        dv_wind = self.tauy / h_vgrid[..., 0, :, 1:-1] * self.space.dy
        du[..., 0, :, :] += du_wind
        dv[..., 0, :, :] += dv_wind
        return du, dv

    def _add_bottom_drag(
        self,
        prognostic: PrognosticTuple,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add bottom drag to the derivatives du, dv.

        Args:
            prognostic (PrognosticTuple): u,v and h.
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with botoom drag forcing.
        """
        coef = self.bottom_drag_coef
        du[..., -1, :, :] += -coef * prognostic.u[..., -1, 1:-1, :]
        dv[..., -1, :, :] += -coef * prognostic.v[..., -1, :, 1:-1]
        return du, dv

    def _create_diagnostic_vars(self, state: State) -> None:
        super()._create_diagnostic_vars(state)

        h_phys = PhysicalLayerDepthAnomaly(ds=self.space.ds)
        U = ZonalVelocityFlux(dx=self.space.dx)  # noqa: N806
        V = MeridionalVelocityFlux(dy=self.space.dy)  # noqa: N806
        omega = Vorticity(masks=self.masks, slip_coef=self.slip_coef)
        eta = SurfaceHeightAnomaly(h_phys=h_phys)
        p = Pressure(g_prime=self.g_prime, eta=eta)
        k_energy = KineticEnergy(masks=self.masks, U=U, V=V)

        U.bind(state)
        V.bind(state)
        omega.bind(state)
        p.bind(state)
        k_energy.bind(state)

    def update(self, prognostic: PrognosticTuple) -> PrognosticTuple:
        """Performs one step time-integration with RK3-SSP scheme.

        Agrs:
            prognostic (PrognosticTuple): u,v and h.
        """
        return schemes.rk3_ssp(
            prognostic,
            self.dt,
            self.compute_time_derivatives,
        )

    def advection_momentum(
        self,
        prognostic: PrognosticTuple,
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
        U = self._state[ZonalVelocityFlux.get_name()].get()  # noqa: N806
        U_m = self.points_to_surfaces(U)  # noqa: N806
        # Meridional velocity -> corresponds to the u grid
        # Has no value on the boundary of the u grid
        V = self._state[MeridionalVelocityFlux.get_name()].get()  # noqa: N806
        V_m = self.points_to_surfaces(V)  # noqa: N806

        # Vortex-force + Coriolis
        omega = self._state[Vorticity.get_name()].get()
        omega_v_m = self._fluxes.w_y(omega[..., 1:-1, :], V_m)
        omega_u_m = self._fluxes.w_x(omega[..., 1:-1], U_m)

        dt_u = omega_v_m + self.fstar_ugrid[..., 1:-1, :] * V_m
        dt_v = -(omega_u_m + self.fstar_vgrid[..., 1:-1] * U_m)

        # grad pressure + k_energy
        k_energy = self._state[KineticEnergy.get_name()].get()
        pressure = self._state[Pressure.get_name()].get()
        ke_pressure = k_energy + pressure
        dt_u -= torch.diff(ke_pressure, dim=-2) + self.dx_p_ref
        dt_v -= torch.diff(ke_pressure, dim=-1) + self.dy_p_ref

        # wind forcing and bottom drag
        dt_u, dt_v = self._add_wind_forcing(dt_u, dt_v)
        dt_u, dt_v = self._add_bottom_drag(prognostic, dt_u, dt_v)

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
        V = self._state[MeridionalVelocityFlux.get_name()].get()  # noqa: N806
        h_tot_flux_y = self._fluxes.h_y(h_tot, V[..., 1:-1])
        # Compute (h_tot x U)
        U = self._state[ZonalVelocityFlux.get_name()].get()  # noqa: N806
        h_tot_flux_x = self._fluxes.h_x(h_tot, U[..., 1:-1, :])
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
        self._state.update_uvh(uvh)
        dt_h = self.advection_h(uvh.h)
        dt_u, dt_v = self.advection_momentum(uvh)
        return UVH(
            dt_u,
            dt_v,
            dt_h,
        )
