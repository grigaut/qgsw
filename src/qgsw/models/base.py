"""Base models class."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.models.core import finite_diff, flux
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.exceptions import (
    IncoherentWithMaskError,
)
from qgsw.models.io import ModelResultsRetriever
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.variables import UVH
from qgsw.spatial.core import grid_conversion
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    import numpy as np

    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


def reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pytorch cumsum in the reverse order.

    Example:
    reverse_cumsum(torch.arange(1,4), dim=-1)
    >>> tensor([6, 5, 3])

    Args:
        x (torch.Tensor): Tensor.
        dim (int): Dimension to perform reverse cumsum on.

    Returns:
        torch.Tensor: Result
    """
    return x + torch.sum(x, dim=dim, keepdim=True) - torch.cumsum(x, dim=dim)


class Model(ModelParamChecker, ModelResultsRetriever, metaclass=ABCMeta):
    """Base class for models.

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

    References variables are denoted with the subscript _ref:
        - h_ref
        - eta_ref
        - p_ref
        - h_ref_ugrid
        - h_ref_vgrid
        - dx_p_ref
        - dy_p_ref
    """

    dtype = torch.float64
    device = DEVICE
    _taux: torch.Tensor | float = 0.0
    _tauy: torch.Tensor | float = 0.0

    omega: torch.Tensor
    eta: torch.Tensor
    U: torch.Tensor
    V: torch.Tensor
    V_m: torch.Tensor
    U_m: torch.Tensor
    h_tot_ugrid: torch.Tensor
    h_tot_vgrid: torch.Tensor

    def __init__(
        self,
        *,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        n_ens: int = 1,
        optimize: bool = True,
    ) -> None:
        """Model Instantiation.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            n_ens (int, optional): Number of ensembles. Defaults to 1.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )
        ModelParamChecker.__init__(
            self,
            space_3d=space_3d,
            g_prime=g_prime,
            beta_plane=beta_plane,
            n_ens=n_ens,
        )
        ModelResultsRetriever.__init__(self)
        self._compute_coriolis()
        ## Topography and Ref values
        self._set_ref_variables()

        # initialize variables
        self.uvh = self._initialize_vars()

        self._set_utils(optimize)
        self._set_fluxes(optimize)

    @property
    def u(self) -> torch.Tensor:
        """State Variable u: Zonal Speed."""
        return self.uvh.u

    @property
    def v(self) -> torch.Tensor:
        """State Variable v: Meridional Speed."""
        return self.uvh.v

    @property
    def h(self) -> torch.Tensor:
        """State Variable h: Layers Thickness."""
        return self.uvh.h

    def _compute_coriolis(
        self,
    ) -> None:
        """Set Coriolis Values."""
        # Coriolis values
        f = self.beta_plane.compute_over_grid(self.space.omega.remove_z_h())
        self.f = f.unsqueeze(0)

    def set_wind_forcing(
        self,
        taux: float | torch.Tensor,
        tauy: float | torch.Tensor,
    ) -> None:
        """Set the winf forcing attributes taux and tauy.

        # TODO: consider implementing validation
        in taux and tauy properties getters/

        Args:
            taux (float | torch.Tensor): Taux value.
            tauy (float | torch.Tensor): Tauy value.
        """
        self._set_taux(taux)
        self._set_tauy(tauy)

    def _set_ref_variables(self) -> None:
        """Set reference variables values.

        Concerned variables:
        - self.h_ref
        - self.eta_ref
        - self.p_ref
        - self.h_ref_ugrid
        - self.h_ref_vgrid
        - self.dx_p_ref
        - self.dy_p_ref
        """
        self.h_ref = self.H * self.space.area
        self.eta_ref = -self.H.sum(dim=-3) + reverse_cumsum(self.H, dim=-3)
        self.p_ref = torch.cumsum(self.g_prime * self.eta_ref, dim=-3)
        if self.h_ref.shape[-2] != 1 and self.h_ref.shape[-1] != 1:
            h_ref_ugrid = F.pad(self.h_ref, (0, 0, 1, 1), mode="replicate")
            self.h_ref_ugrid = 0.5 * (
                h_ref_ugrid[..., 1:, :] + h_ref_ugrid[..., :-1, :]
            )
            h_ref_vgrid = F.pad(self.h_ref, (1, 1), mode="replicate")
            self.h_ref_vgrid = 0.5 * (
                h_ref_vgrid[..., 1:] + h_ref_vgrid[..., :-1]
            )
            self.dx_p_ref = torch.diff(self.p_ref, dim=-2)
            self.dy_p_ref = torch.diff(self.p_ref, dim=-1)
        else:
            self.h_ref_ugrid = self.h_ref
            self.h_ref_vgrid = self.h_ref
            self.dx_p_ref = 0.0
            self.dy_p_ref = 0.0

    def _set_utils(self, optimize: bool) -> None:  # noqa: FBT001
        """Set utils functions.

        Args:
            optimize (bool): Whether to optimize the function.
        """
        if optimize:
            self.comp_ke = OptimizableFunction(finite_diff.comp_ke)
            self.cell_corners_to_cell_centers = OptimizableFunction(
                grid_conversion.cell_corners_to_cell_center,
            )
        else:
            self.comp_ke = finite_diff.comp_ke
            self.cell_corners_to_cell_centers = (
                grid_conversion.cell_corners_to_cell_center
            )

    def _set_fluxes(self, optimize: bool) -> None:  # noqa: FBT001
        """Set fluxes.

        Args:
            optimize (bool): Whether to optimize the fluxes.
        """
        self._fluxes = flux.Fluxes(masks=self.masks, optimize=optimize)

    def _initialize_vars(self) -> UVH:
        """Initialize variables.

        Create Empty variables.

        Concerned variables:
        - u
        - v
        - h
        """
        base_shape = (self.n_ens, self.space.nl)
        h = torch.zeros(
            (*base_shape, self.space.nx, self.space.ny),
            dtype=self.dtype,
            device=self.device,
        )
        u = torch.zeros(
            (*base_shape, self.space.nx + 1, self.space.ny),
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.zeros(
            (*base_shape, self.space.nx, self.space.ny + 1),
            dtype=self.dtype,
            device=self.device,
        )
        return UVH(u, v, h)

    def compute_omega(self, uvh: UVH) -> torch.Tensor:
        """Pad u, v using boundary conditions.

        Possible boundary conditions: free-slip, partial free-slip, no-slip.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: result
        """
        u_ = F.pad(uvh.u, (1, 1, 0, 0))
        v_ = F.pad(uvh.v, (0, 0, 1, 1))
        dx_v = torch.diff(v_, dim=-2)
        dy_u = torch.diff(u_, dim=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self.slip_coef)
        omega: torch.Tensor = (
            self.masks.w_valid * curl_uv
            + self.masks.w_cornerout_bound * (1 - self.slip_coef) * curl_uv
            + self.masks.w_vertical_bound * alpha * dx_v
            - self.masks.w_horizontal_bound * alpha * dy_u
        )
        return omega

    def compute_diagnostic_variables(self, uvh: UVH) -> None:
        """Compute the model's diagnostic variables.

        Args:
            uvh (UVH): Prognostic variables.

        Computed variables:
        - Vorticity: omega
        - Interface heights: eta
        - Pressure: p
        - Zonal velocity: U
        - Meridional velocity: V
        - Zonal Velocity Momentum: U_m
        - Meriodional Velocity Momentum: V_m
        - Kinetic Energy: k_energy

        Compute the result given the prognostic
        variables self.u, self.v, self.h .
        """
        # Diagnostic: vorticity values
        self.omega = self.compute_omega(uvh)
        # Diagnostic: interface height : physical
        self.eta = reverse_cumsum(self.h / self.space.area, dim=-3)
        # Diagnostic: pressure values
        self.p = torch.cumsum(self.g_prime * self.eta, dim=-3)
        # Diagnostic: zonal velocity
        self.U = uvh.u / self.space.dx**2
        # Diagnostic: meridional velocity
        self.V = uvh.v / self.space.dy**2
        # Zonal velocity momentum -> corresponds to the v grid
        # Has no value on the boundary of the v grid
        self.U_m = self.cell_corners_to_cell_centers(self.U)
        # Meridional velocity momentum -> corresponds to the u grid
        # Has no value on the boundary of the u grid
        self.V_m = self.cell_corners_to_cell_centers(self.V)
        # Diagnostic: kinetic energy
        self.k_energy = (
            self.comp_ke(uvh.u, self.U, uvh.v, self.V) * self.masks.h
        )
        # Match u grid dimensions (1, nl, nx, ny)
        self.h_ugrid = grid_conversion.h_to_u(uvh.h, self.masks.h)
        # Match v grid dimension
        self.h_vgrid = grid_conversion.h_to_v(uvh.h, self.masks.h)
        # Sum h on u grid
        self.h_tot_ugrid = self.h_ref_ugrid + self.h_ugrid
        # Sum h on v grid
        self.h_tot_vgrid = self.h_ref_vgrid + self.h_vgrid

    @abstractmethod
    def set_physical_uvh(
        self,
        u_phys: torch.Tensor | np.ndarray,
        v_phys: torch.Tensor | np.ndarray,
        h_phys: torch.Tensor | np.ndarray,
    ) -> None:
        """Set state variables from physical variables.

        Args:
            u_phys (torch.Tensor|np.ndarray): Physical U.
            v_phys (torch.Tensor|np.ndarray): Physical V.
            h_phys (torch.Tensor|np.ndarray): Physical H.
        """

    def set_uvh(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Set u,v,h value from state variables.

        Warning: the expected values are not physical values but state values.
        The variables correspond to the actual self.u, self.v, self.h
        of the model.

        Args:
            u (torch.Tensor): State variable u.
            v (torch.Tensor): State variable v.
            h (torch.Tensor): State variable h.
        """
        u = u.to(self.device)
        v = v.to(self.device)
        h = h.to(self.device)

        if not (u * self.masks.u == u).all():
            msg = (
                "Input velocity u incoherent with domain mask, "
                "velocity must be zero out of domain."
            )
            raise IncoherentWithMaskError(msg)

        if not (v * self.masks.v == v).all():
            msg = (
                "Input velocity v incoherent with domain mask, "
                "velocity must be zero out of domain."
            )
            raise IncoherentWithMaskError(msg)
        u = u.type(self.dtype) * self.masks.u
        v = v.type(self.dtype) * self.masks.v
        h = h.type(self.dtype) * self.masks.h
        self.uvh = UVH(u, v, h)
        self.compute_diagnostic_variables(self.uvh)

    @abstractmethod
    def advection_h(self) -> torch.Tensor:
        """Advection RHS for thickness perturbation h.

        dt_h = - div(h_tot [u v])

        h_tot = h_ref + h

        Returns:
            torch.Tensor: h advection.
        """

    @abstractmethod
    def advection_momentum(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advection RHS for momentum (u, v).

        Returns:
            tuple[torch.Tensor,torch.Tensor]: u, v advection.
        """

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
        h_ugrid = self.h_tot_ugrid / self.space.area
        h_vgrid = self.h_tot_vgrid / self.space.area
        du_wind = self.taux / h_ugrid[..., 0, 1:-1, :] * self.space.dx
        dv_wind = self.tauy / h_vgrid[..., 0, :, 1:-1] * self.space.dy
        du[..., 0, :, :] += du_wind
        dv[..., 0, :, :] += dv_wind
        return du, dv

    def _add_bottom_drag(
        self,
        du: torch.Tensor,
        dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add bottom drag to the derivatives du, dv.

        Args:
            du (torch.Tensor): du
            dv (torch.Tensor): dv

        Returns:
            tuple[torch.Tensor, torch.Tensor]: du, dv with botoom drag forcing.
        """
        du[..., -1, :, :] += -self.bottom_drag_coef * self.u[..., -1, 1:-1, :]
        dv[..., -1, :, :] += -self.bottom_drag_coef * self.v[..., -1, :, 1:-1]
        return du, dv

    @abstractmethod
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

    @abstractmethod
    def update(self, uvh: UVH) -> UVH:
        """Update prognostic variables.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: update prognostic variables.
        """

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self.uvh = self.update(self.uvh)
