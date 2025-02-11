"""Quasi Geostrophic Model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from qgsw.fields.variables.uvh import UVH, UVHT, BasePrognosticTuple
from qgsw.models.base import Model
from qgsw.models.core import schemes
from qgsw.models.exceptions import UnsetTimestepError
from qgsw.models.names import ModelName
from qgsw.models.qg.projectors.core import QGProjector
from qgsw.models.qg.stretching_matrix import (
    compute_A,
)
from qgsw.models.qg.variable_set import QGVariableSet
from qgsw.models.sw.core import SW
from qgsw.spatial.core.discretization import SpaceDiscretization2D

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.masks import Masks
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization2D


T = TypeVar("T", bound=BasePrognosticTuple)
Projector = TypeVar("Projector", bound=QGProjector)


class QGCore(Model[T], Generic[T, Projector]):
    """Quasi Geostrophic Model."""

    _save_p_values = False

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """QG Model Instantiation.

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
        self.A = self.compute_A(H, g_prime)
        self._set_projector()
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )

    @property
    def save_p_values(self) -> bool:
        """Whether to save pressure values from integration steps or not."""
        return self._save_p_values

    @property
    def intermediate_p_values(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Intermediate pressure values.

        Raises:
            AttributeError: If the model does not save pressure values.
        """
        if self.save_p_values:
            return self._p_vals
        msg = (
            "The model does not save intermediate p values."
            "Call .save_intermediate_p() to access this attribute"
        )
        raise AttributeError(msg)

    @property
    def P(self) -> Projector:  # noqa: N802
        """QG projector."""
        return self._P

    @property
    def sw(self) -> SW:
        """Core Shallow Water Model."""
        return self._core

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
    def masks(self, masks: Masks) -> None:
        """Masks setter."""
        Model.masks.fset(self, masks)
        self.sw.masks = masks
        self._P.masks = masks

    @Model.n_ens.setter
    def n_ens(self, n_ens: int) -> None:
        """Ensemble number setter."""
        Model.n_ens.fset(self, n_ens)
        self.sw.n_ens = n_ens

    @property
    def uvh_dt(self) -> UVH:
        """Prognostic variable increment from previous iteration.

        ├── u_dt: (n_ens, nl, nx+1, ny)-shaped
        ├── v_dt: (n_ens, nl, nx, ny+1)-shaped
        └── h_dt: (n_ens, nl, nx, ny)-shaped
        """
        return self._uvh_dt

    def save_intermediate_p(self) -> None:
        """Save intermediate pressure values from P."""
        self._save_p_values = True

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
        ]
        try:
            msg_parts.append(f"├── dt: {self.dt} s")
        except UnsetTimestepError:
            msg_parts.append("├── dt: unset yet")
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
                └── (nl,)-shaped
            g_prime (torch.Tensor): Reduced gravity values.
                └── (nl,)-shaped

        Returns:
            torch.Tensor: Stretching Operator.
                └── (nl, nl)-shaped
        """
        return compute_A(
            H=H,
            g_prime=g_prime,
            dtype=self.dtype,
            device=self.device.get(),
        )

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
                └── (n_ens, nl, nx+1, ny)-shaped
            v (torch.Tensor): State variable v.
                └── (n_ens, nl, nx, ny+1)-shaped
            h (torch.Tensor): State variable h.
                └── (n_ens, nl, nx, ny)-shaped
        """
        self.sw.set_uvh(u, v, h)
        super().set_uvh(u, v, h)

    def _set_projector(self) -> None:
        """Set the projector."""
        self._P = QGProjector(
            self.A,
            self.H,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

    def _init_core_model(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool,  # noqa: FBT001
    ) -> SW:
        """Initialize the core Shallow Water model.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.

        Returns:
            SW: Core model.
        """
        return SW(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )

    def compute_time_derivatives(
        self,
        uvh: UVH,
    ) -> UVH:
        """Compute the prognostic variables derivatives dt_u, dt_v, dt_h.

        Args:
            uvh (UVH): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: dt_u, dt_v, dt_h
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        dt_prognostic_sw = self.sw.compute_time_derivatives(uvh)
        self._uvh_dt = self._P.project(dt_prognostic_sw)
        if self.save_p_values:
            p, p_i = self.P.compute_p(dt_prognostic_sw)
            self._p_vals.append((p, p_i))
        return self._uvh_dt

    def update(self, uvh: UVH) -> UVH:
        """Update uvh.

        Args:
            uvh (UVH): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: update prognostic variables.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        if self.save_p_values:
            self._p_vals = []
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
        return QGVariableSet.get_variable_set(space, physics, model)


class QG(QGCore[UVHT, QGProjector]):
    """Quasi Geostrophic Model."""

    _type = ModelName.QUASI_GEOSTROPHIC
