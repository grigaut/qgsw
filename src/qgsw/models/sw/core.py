"""Shallow-water implementation.

Louis Thiry, Nov 2023 for IFREMER.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.fields.variables.covariant import (
    KineticEnergy,
    MaskedVorticity,
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    Pressure,
    PressureTilde,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.state import BaseStateUVH, StateUVH, StateUVHAlpha
from qgsw.fields.variables.tuples import (
    UVH,
    UVHT,
    BaseUVH,
    UVHTAlpha,
)
from qgsw.models.base import ModelUVH
from qgsw.models.core import finite_diff, schemes
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sw.variable_set import SWVariableSet
from qgsw.spatial.core import grid_conversion as convert
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D
    from qgsw.spatial.core.grid import Grid2D


def inv_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of reverse cumsum function.

    Args:
        x (torch.Tensor): Tensor to perform inverse reverse cumsum on.
        dim (int): Dimension to perform the inverse reverse cumsum on.

    Returns:
        torch.Tensor: Inverse Reverse Cumsum
    """
    return torch.cat([-torch.diff(x, dim=dim), x.narrow(dim, -1, 1)], dim=dim)


T = TypeVar("T", bound=BaseUVH)
State = TypeVar("State", bound=BaseStateUVH)


class SWCore(ModelUVH[T, State], Generic[T, State]):
    """Implementation of multilayer rotating shallow-water model.

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
      - eta_phys = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """

    _type = ModelName.SHALLOW_WATER
    _pressure_name: str

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,
    ) -> None:
        """SW Model Instantiation.

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

    @property
    def P(self) -> QGProjector:  # noqa: N802
        """Quasi-Geostrophic projector."""
        try:
            return self._P
        except AttributeError:
            self._set_projector()
            return self.P

    def _compute_coriolis(self, omega_grid_2d: Grid2D) -> None:
        """Set Coriolis related grids.

        Args:
            omega_grid_2d (Grid2D): Omega grid (2D).
        """
        super()._compute_coriolis(omega_grid_2d=omega_grid_2d)
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
            prognostic (UVHT): u,v and h.
            du (torch.Tensor): du
                └── (n_ens, nl, nx+1, ny)-shaped
            dv (torch.Tensor): dv
                └── (n_ens, nl, nx, ny+1)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with wind forcing.
                ├── du: (n_ens, nl, nx+1, ny)-shaped
                └── dv: (n_ens, nl, nx, ny+1)-shaped
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
        prognostic: UVHT,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add bottom drag to the derivatives du, dv.

        Args:
            prognostic (UVHT): u,v and h.
            du (torch.Tensor): du
                └── (n_ens, nl, nx+1, ny)-shaped
            dv (torch.Tensor): dv
                └── (n_ens, nl, nx, ny+1)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with botoom drag forcing.
                ├── du: (n_ens, nl, nx+1, ny)-shaped
                └── dv: (n_ens, nl, nx, ny+1)-shaped
        """
        coef = self.bottom_drag_coef
        du[..., -1, :, :] += -coef * prognostic.u[..., -1, 1:-1, :]
        dv[..., -1, :, :] += -coef * prognostic.v[..., -1, :, 1:-1]
        return du, dv

    def _create_diagnostic_vars(self, state: State) -> None:
        """Create diagnostic variables and bind them to state.

        Args:
            state (State): state.
        """
        super()._create_diagnostic_vars(state)

        U = ZonalVelocityFlux(dx=self.space.dx)  # noqa: N806
        V = MeridionalVelocityFlux(dy=self.space.dy)  # noqa: N806
        omega = MaskedVorticity(masks=self.masks, slip_coef=self.slip_coef)
        h_phys = self._state[PhysicalLayerDepthAnomaly.get_name()]
        eta_phys = PhysicalSurfaceHeightAnomaly(h_phys=h_phys)
        p = Pressure(g_prime=self.g_prime, eta_phys=eta_phys)
        self._pressure_name = p.get_name()
        k_energy = KineticEnergy(masks=self.masks, U=U, V=V)

        U.bind(state)
        V.bind(state)
        omega.bind(state)
        p.bind(state)
        k_energy.bind(state)

    def _set_projector(self) -> None:
        """Set the projector."""
        self._P = QGProjector(
            compute_A(
                self.H[:, 0, 0],
                self.g_prime[:, 0, 0],
                dtype=self.dtype,
                device=self.device.get(),
            ),
            self.H,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

    def update(self, prognostic: UVH) -> UVH:
        """Performs one step time-integration with RK3-SSP scheme.

        Agrs:
            prognostic (UVH): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return schemes.rk3_ssp(
            prognostic,
            self.dt,
            self.compute_time_derivatives,
        )

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        uvh = QGProjector.G(
            p,
            compute_A(
                self.H[:, 0, 0],
                self.g_prime[:, 0, 0],
                dtype=torch.float64,
                device=DEVICE.get(),
            ),
            self.H,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.points_to_surfaces,
        )
        self.set_uvh(*uvh)

    def advection_momentum(
        self,
        prognostic: UVHT,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Use the shallow water momentum equation in rotation frame:
        ∂_t [u v] + (ω + f)x[u v] = -∇(gη + 0.5[u v]⋅[u v])
        Therefore:
        - ∂_t u = (ω + f)v - ∂_x(gη + 0.5(u² + v²))
        - ∂_t v = - (ω + f)u - ∂_y(gη + 0.5(u² + v²))

        Agrs:
            prognostic (UVHT): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection (∂_t u, ∂_t v).
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                └── v: (n_ens, nl, nx, ny+1)-shaped
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
        omega = self._state[MaskedVorticity.get_name()].get()
        omega_v_m = self._fluxes.w_y(omega[..., 1:-1, :], V_m)
        omega_u_m = self._fluxes.w_x(omega[..., 1:-1], U_m)

        dt_u = omega_v_m + self.fstar_ugrid[..., 1:-1, :] * V_m
        dt_v = -(omega_u_m + self.fstar_vgrid[..., 1:-1] * U_m)

        # grad pressure + k_energy
        k_energy = self._state[KineticEnergy.get_name()].get()
        pressure = self._state[self._pressure_name].get()
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
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: h advection (∂_t h)
                └── (n_ens, nl, nx, ny)-shaped
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
            uvh (UVH): u,v and h
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: dt_u, dt_v, dt_h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        self._state.update_uvh(uvh)
        dt_h = self.advection_h(uvh.h)
        dt_u, dt_v = self.advection_momentum(uvh)
        return UVH(
            dt_u,
            dt_v,
            dt_h,
        )

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,
        physics: PhysicsConfig,
        model: ModelConfig,
    ) -> dict[str, DiagnosticVariable]:
        """Create variable set.

        Args:
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
            model (ModelConfig): Model configuaration.

        Returns:
            dict[str, DiagnosticVariable]: Variables dictionnary.
        """
        return SWVariableSet.get_variable_set(space, physics, model)


class SW(SWCore[UVHT, StateUVH]):
    """Implementation of multilayer rotating shallow-water model.

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
      - eta_phys = eta_phys                  (interface height)
      - p = p_phys                      (hydrostratic pressure)
      - k_energy = k_energy_phys        (kinetic energy)
      - pv = pv_phys                    (potential vorticity)

    ## Time integration
    Explicit time integration with RK3-SSP scheme.

    """


class SWCollinearSublayer(SWCore[UVHTAlpha, StateUVHAlpha]):
    """Shallow water for collinear sublayer models."""

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,
    ) -> None:
        """SW Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        self.__instance_nb = next(self._instance_count)
        self.name = f"{self.__class__.__name__}-{self.__instance_nb}"
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
        )
        self._space = keep_top_layer(self._space)

        self._compute_coriolis(self._space.omega.remove_z_h())
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._set_state()
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H[:1, ...]

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)

    def _create_diagnostic_vars(self, state: StateUVHAlpha) -> None:
        self._state.unbind()
        self._create_physical_variables(state)

        U = ZonalVelocityFlux(dx=self.space.dx)  # noqa: N806
        V = MeridionalVelocityFlux(dy=self.space.dy)  # noqa: N806
        omega = MaskedVorticity(masks=self.masks, slip_coef=self.slip_coef)
        h_phys = self._state[PhysicalLayerDepthAnomaly.get_name()]
        eta_phys = PhysicalSurfaceHeightAnomaly(h_phys=h_phys)
        p = PressureTilde(g_prime=self.g_prime, eta_phys=eta_phys)
        self._pressure_name = p.get_name()
        k_energy = KineticEnergy(masks=self.masks, U=U, V=V)

        U.bind(state)
        V.bind(state)
        omega.bind(state)
        p.bind(state)
        k_energy.bind(state)

    def _set_io(self, state: StateUVHAlpha) -> None:
        self._io = IO(
            state.t,
            state.alpha,
            *state.physical,
        )

    def _set_state(self) -> None:
        self._state = StateUVHAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
