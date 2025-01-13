"""Base models class."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import UVH, BasePrognosticTuple
from qgsw.models.core import finite_diff, flux
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.exceptions import (
    IncoherentWithMaskError,
)
from qgsw.models.io import IO
from qgsw.models.parameters import ModelParamChecker
from qgsw.spatial.core import grid_conversion as convert
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D
    from qgsw.specs._utils import Device

Prognostic = TypeVar("Prognostic", bound=BasePrognosticTuple)


class Model(ModelParamChecker, Generic[Prognostic], metaclass=ABCMeta):
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

    _type: str

    dtype = torch.float64
    device: Device = DEVICE
    _taux: torch.Tensor | float = 0.0
    _tauy: torch.Tensor | float = 0.0

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
        )
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._set_state()
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)

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
        state_repr = ["└── " + state_repr_.pop(0)]
        state_repr = state_repr + ["\t" + txt for txt in state_repr_]
        return msg_parts + space_repr + state_repr

    def __repr__(self) -> str:
        """String representation of the model."""
        return "\n[Model]\n" + "\n".join(self.get_repr_parts())

    @property
    def io(self) -> IO:
        """Input/Output manager."""
        return self._io

    @property
    def prognostic(self) -> Prognostic:
        """Prognostic tuple."""
        return self._state.prognostic

    @property
    def u(self) -> torch.Tensor:
        """State Variable u: Zonal Speed."""
        return self._state.u.get()

    @property
    def v(self) -> torch.Tensor:
        """State Variable v: Meridional Speed."""
        return self._state.v.get()

    @property
    def h(self) -> torch.Tensor:
        """State Variable h: Layers Thickness."""
        return self._state.h.get()

    @ModelParamChecker.slip_coef.setter
    def slip_coef(self, slip_coef: float) -> None:
        """Slip coefficient setter."""
        ModelParamChecker.slip_coef.fset(self, slip_coef)
        self._create_diagnostic_vars(self._state)

    @ModelParamChecker.beta_plane.setter
    def beta_plane(self, beta_plane: BetaPlane) -> None:
        """Beta plane setter."""
        ModelParamChecker.beta_plane.fset(self, beta_plane)
        self._compute_coriolis()

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
        self.h_ref = self.H * self.space.ds
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

    def _set_state(self) -> None:
        self._state = State.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self._io = IO(
            t=self._state.t,
            u=self._state.u,
            v=self._state.v,
            h=self._state.h,
        )

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
            h_ * self.space.ds,
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
        self._state.update_uvh(UVH(u, v, h))

    @abstractmethod
    def compute_time_derivatives(
        self,
        prognostic: UVH,
    ) -> UVH:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Args:
            prognostic (UVH): u,v and h.

        Returns:
            UVH: dt_u, dt_v, dt_h
        """

    @abstractmethod
    def update(self, uvh: UVH) -> UVH:
        """Update u,v and h.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            UVH: update prognostic variables.
        """

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.increment_time(self.dt)
        self._state.update_uvh(self.update(self._state.prognostic.uvh))

    @classmethod
    def get_type(cls) -> str:
        """Get the model type.

        Returns:
            str: Model type.
        """
        return cls._type
