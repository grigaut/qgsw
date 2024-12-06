"""Base models class."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.models.core import finite_diff, flux
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.exceptions import (
    IncoherentWithMaskError,
)
from qgsw.models.io import ModelResultsRetriever
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.variables import (
    UVH,
    KineticEnergy,
    Pressure,
    State,
    SurfaceHeightAnomaly,
    Vorticity,
)
from qgsw.models.variables.dynamics import Momentum
from qgsw.spatial.core import grid_conversion as convert
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization3D
    from qgsw.specs._utils import Device


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
    device: Device = DEVICE
    _taux: torch.Tensor | float = 0.0
    _tauy: torch.Tensor | float = 0.0

    def __init__(
        self,
        *,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
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
        )
        ModelResultsRetriever.__init__(self)
        self._compute_coriolis()
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._state = State.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)

    @property
    def uvh(self) -> UVH:
        """UVH."""
        return self._state.uvh

    @property
    def u(self) -> torch.Tensor:
        """State Variable u: Zonal Speed."""
        return self._state.u

    @property
    def v(self) -> torch.Tensor:
        """State Variable v: Meridional Speed."""
        return self._state.v

    @property
    def h(self) -> torch.Tensor:
        """State Variable h: Layers Thickness."""
        return self._state.h

    @property
    def omega(self) -> torch.Tensor:
        """Vorticity."""
        return self._omega.get()

    @property
    def eta(self) -> torch.Tensor:
        """Surface height anomaly."""
        return self._eta.get()

    @property
    def p(self) -> torch.Tensor:
        """Pressure."""
        return self._p.get()

    @property
    def U(self) -> torch.Tensor:  # noqa: N802
        """Momentum on X.."""
        return self._UV.get()[0]

    @property
    def V(self) -> torch.Tensor:  # noqa: N802
        """Momentum on Y."""
        return self._UV.get()[1]

    @property
    def k_energy(self) -> torch.Tensor:
        """Physical meriodional velocity."""
        return self._k_energy.get()

    @ModelParamChecker.slip_coef.setter
    def slip_coef(self, slip_coef: float) -> None:
        """Timestep setter."""
        ModelParamChecker.slip_coef.fset(self, slip_coef)
        self._create_diagnostic_vars(self._state)

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
        """Set the wind forcing attributes taux and tauy.

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
            self.points_to_surfaces = OptimizableFunction(
                convert.points_to_surfaces,
            )
        else:
            self.comp_ke = finite_diff.comp_ke
            self.points_to_surfaces = convert.points_to_surfaces

    def _set_fluxes(self, optimize: bool) -> None:  # noqa: FBT001
        """Set fluxes.

        Args:
            optimize (bool): Whether to optimize the fluxes.
        """
        self._fluxes = flux.Fluxes(masks=self.masks, optimize=optimize)

    def _create_diagnostic_vars(self, state: State) -> None:
        state.unbind()
        UV = Momentum(dx=self.space.dx, dy=self.space.dy)  # noqa: N806
        omega = Vorticity(UV=UV, masks=self.masks, slip_coef=self.slip_coef)
        eta = SurfaceHeightAnomaly(area=self.space.area)
        p = Pressure(g_prime=self.g_prime, eta=eta)
        k_energy = KineticEnergy(masks=self.masks, UV=UV)

        self._omega = omega.bind(state)
        self._eta = eta.bind(state)
        self._p = p.bind(state)
        self._UV = UV.bind(state)
        self._k_energy = k_energy.bind(state)

    def set_physical_uvh(
        self,
        u_phys: torch.Tensor | np.ndarray,
        v_phys: torch.Tensor | np.ndarray,
        h_phys: torch.Tensor | np.ndarray,
    ) -> None:
        """Set state variables from physical variables.

        As a reminder, the physical variables u_phys, v_phys, h_phys
        are linked to the state variable u,v,h through:
        - u = u_phys * dx
        - v = v_phys * dy
        - h = h_phys * dx * dy

        Args:
            u_phys (torch.Tensor|np.ndarray): Physical U.
            v_phys (torch.Tensor|np.ndarray): Physical V.
            h_phys (torch.Tensor|np.ndarray): Physical H.
        """
        u_ = (
            torch.from_numpy(u_phys)
            if isinstance(u_phys, np.ndarray)
            else u_phys
        )
        v_ = (
            torch.from_numpy(v_phys)
            if isinstance(v_phys, np.ndarray)
            else v_phys
        )
        h_ = (
            torch.from_numpy(h_phys)
            if isinstance(h_phys, np.ndarray)
            else h_phys
        )
        self.set_uvh(
            u_ * self.space.dx,
            v_ * self.space.dy,
            h_ * self.space.area,
        )

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
        u = u.to(self.device.get())
        v = v.to(self.device.get())
        h = h.to(self.device.get())

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
        self._state.update(u, v, h)

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
        self._state.uvh = self.update(self._state.uvh)
