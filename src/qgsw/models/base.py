"""Base models class."""

from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.exceptions import (
    IncoherentWithMaskError,
    UnsetTimestepError,
)
from qgsw.fields.variables.prognostic_tuples import (
    PSIQ,
    UVH,
    BasePrognosticPSIQ,
    BasePrognosticTuple,
    BasePrognosticUVH,
)
from qgsw.fields.variables.state import BaseState, BaseStateUVH, StateUVH
from qgsw.models.core import finite_diff, flux
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.io import IO
from qgsw.models.names import ModelCategory, ModelName, get_category
from qgsw.models.parameters import ModelParamChecker
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core import grid_conversion as convert
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import DEVICE
from qgsw.utils.named_object import NamedObject

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.models.qg.uvh.projectors.core import QGProjector
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D
    from qgsw.spatial.core.grid import Grid2D
    from qgsw.specs._utils import Device

Prognostic = TypeVar("Prognostic", bound=BasePrognosticTuple)
AdvectedPrognostic = TypeVar("AdvectedPrognostic", bound=Union[UVH, PSIQ])
State = TypeVar("State", bound=BaseState)
PrognosticUVH = TypeVar("PrognosticUVH", bound=BasePrognosticUVH)
PrognosticPSIQ = TypeVar("PrognosticPSIQ", bound=BasePrognosticPSIQ)


class ModelCounter(type):
    """Metaclass to make instance counter not share count with descendants."""

    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict) -> None:
        """Model Counter metaclass."""
        super().__init__(name, bases, attrs)
        cls._instance_count = itertools.count(1)


class ABCCounter(ModelCounter, ABCMeta):
    """Multiple inheritance for counter."""


class _Model(
    ModelParamChecker,
    Generic[Prognostic, State, AdvectedPrognostic],
    NamedObject[ModelName],
    metaclass=ABCCounter,
):
    """Base class for models."""

    _state: State
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
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: ARG002
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity tensor.
                └── (nl,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        self.__instance_nb = next(self._instance_count)
        self.__name = f"{self.__class__.__name__}-{self.__instance_nb}"
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
        self._compute_coriolis(self._space.omega.remove_z_h())

    @property
    def courant_number(self) -> float:
        """Courant number: v*dt/dx."""
        return (
            torch.sqrt(torch.sum(self.H) * self.g_prime[0])
            * self.dt
            / torch.min(self.space.dx, self.space.dy)
        ).item()

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        msg_parts = [
            f"Model: {self.__class__}",
            f"├── Data type: {self.dtype}",
            f"├── Device: {self.device}",
            f"├── Courant number: {self.courant_number}",
            (
                f"├── Beta plane: f0 = {self.beta_plane.f0} "
                f"- β = {self.beta_plane.beta}"
            ),
        ]
        try:
            msg_parts.append(f"├── dt: {self.dt} s")
        except UnsetTimestepError:
            msg_parts.append("├── dt: unset yet")
        space_repr_ = self.space.get_repr_parts()
        space_repr = ["├── " + space_repr_.pop(0)]
        space_repr = space_repr + ["│\t" + txt for txt in space_repr_]
        state_repr_ = self._state.get_repr_parts()
        state_repr = ["└── " + state_repr_.pop(0)]
        state_repr = state_repr + ["\t" + txt for txt in state_repr_]
        return msg_parts + space_repr + state_repr

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"\n[Model] -> `{self.name}`\n" + "\n".join(
            self.get_repr_parts(),
        )

    @property
    def time(self) -> torch.Tensor:
        """Model time."""
        return self._state.t.get()

    @property
    def name(self) -> str:
        """Object name."""
        try:
            return self._name
        except AttributeError:
            return self.__name

    @name.setter
    def name(self, name: str) -> None:
        if name is None:
            return
        self._name = name

    @property
    def io(self) -> IO:
        """Input/Output manager."""
        return self._io

    @property
    def prognostic(self) -> Prognostic:
        """Prognostic tuple."""
        return self._state.prognostic

    @ModelParamChecker.slip_coef.setter
    def slip_coef(self, slip_coef: float) -> None:
        """Slip coefficient setter."""
        ModelParamChecker.slip_coef.fset(self, slip_coef)
        self._create_diagnostic_vars(self._state)

    def _compute_coriolis(
        self,
        omega_grid_2d: Grid2D,
    ) -> None:
        """Set Coriolis Values.

        Args:
            omega_grid_2d (Grid2D): Omega grid (2D).
        """
        # Coriolis values
        f = self.beta_plane.compute_over_grid(omega_grid_2d)
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

    @abstractmethod
    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """

    @abstractmethod
    def _set_state(self) -> None:
        """Set the state."""

    @abstractmethod
    def _set_utils(self, optimize: bool) -> None:  # noqa: FBT001
        """Set utils functions.

        Args:
            optimize (bool): Whether to optimize the function.
        """

    def _create_diagnostic_vars(self, state: State) -> None:
        state.unbind()

    @abstractmethod
    def compute_time_derivatives(
        self,
        prognostic: AdvectedPrognostic,
    ) -> AdvectedPrognostic:
        """Compute the state variables derivatives dt_u, dt_v, dt_h.

        Args:
            prognostic (UVH): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: dt_u, dt_v, dt_h
                ├── dt_u: (n_ens, nl, nx+1, ny)-shaped
                ├── dt_v: (n_ens, nl, nx, ny+1)-shaped
                └── dt_h: (n_ens, nl, nx, ny)-shaped
        """

    @abstractmethod
    def update(self, prognostic: AdvectedPrognostic) -> AdvectedPrognostic:
        """Update prognostic tuple.

        Args:
            prognostic (AdvectedPrognostic): Prognostic variable to advect.

        Returns:
            AdvectedPrognostic: Updated prognostic variable to advect.
        """

    @abstractmethod
    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        self._state.increment_time(self.dt)

    @classmethod
    @abstractmethod
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

    @classmethod
    def get_category(cls) -> ModelCategory:
        """Get the model category.

        Returns:
            ModelCategory: Model category.
        """
        return get_category(cls.get_type())


State_uvh = TypeVar("State_uvh", bound=BaseStateUVH)


class ModelUVH(
    _Model[PrognosticUVH, State_uvh, UVH],
    Generic[PrognosticUVH, State_uvh],
):
    """Base class for UVH models.

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
        - eta_phys = eta_phys                  (interface height)
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

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (nl,) shaped.
            g_prime (torch.Tensor): Reduced gravity tensor.
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
        # initialize state
        self._set_state()
        # initialize variables
        self._create_diagnostic_vars(self._state)
        self._set_utils(optimize)

        # Ref values
        self._set_ref_variables()
        self._set_fluxes(optimize)

    @property
    def u(self) -> torch.Tensor:
        """StateUVH Variable u: Zonal Speed.

        └── (n_ens, nl, nx+1,ny)-shaped.
        """
        return self._state.u.get()

    @property
    def v(self) -> torch.Tensor:
        """StateUVH Variable v: Meridional Speed.

        └── (n_ens, nl, nx,ny+1)-shaped.
        """
        return self._state.v.get()

    @property
    def h(self) -> torch.Tensor:
        """StateUVH Variable h: Layers Thickness.

        └── (n_ens, nl, nx,ny)-shaped.
        """
        return self._state.h.get()

    @property
    @abstractmethod
    def P(self) -> QGProjector:  # noqa: N802
        """QG Projector."""

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StateUVH.steady(
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

    def _set_fluxes(self, optimize: bool) -> None:  # noqa: FBT001
        """Set fluxes.

        Args:
            optimize (bool): Whether to optimize the fluxes.
        """
        self._fluxes = flux.Fluxes(masks=self.masks, optimize=optimize)

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
                └── (n_ens, nl, nx+1, ny)-shaped.
            v_phys (torch.Tensor|np.ndarray): Physical V.
                └── (n_ens, nl, nx, ny+1)-shaped.
            h_phys (torch.Tensor|np.ndarray): Physical H.
                └── (n_ens, nl, nx, ny)-shaped.
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
            u (torch.Tensor): StateUVH variable u.
                └── (n_ens, nl, nx+1, ny)-shaped.
            v (torch.Tensor): StateUVH variable v.
                └── (n_ens, nl, nx, ny+1)-shaped.
            h (torch.Tensor): StateUVH variable h.
                └── (n_ens, nl, nx, ny)-shaped.
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

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        super().step()
        self._state.update_uvh(self.update(self._state.prognostic.uvh))
