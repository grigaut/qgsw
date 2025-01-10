"""Base classes for variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    Time,
    ZonalVelocity,
)
from qgsw.fields.variables.uvh import UVH, UVHT, UVHTAlpha

if TYPE_CHECKING:
    import torch

    from qgsw.fields.variables.base import (
        BoundDiagnosticVariable,
        PrognosticVariable,
    )


T = TypeVar("T", bound=Union[UVHT, UVHTAlpha])


class BaseState(ABC, Generic[T]):
    """Base State class."""

    def __init__(self, prognostic: T) -> None:
        """Instantiate State.

        Args:
            prognostic (T): Prognostic variables.
        """
        self.unbind()
        self._prog = prognostic
        self._t = Time(prognostic.t)
        self._u = ZonalVelocity(prognostic.u)
        self._v = MeridionalVelocity(prognostic.v)
        self._h = LayerDepthAnomaly(prognostic.h)
        self._prog_vars = {
            Time.get_name(): self._t,
            ZonalVelocity.get_name(): self._u,
            MeridionalVelocity.get_name(): self._v,
            LayerDepthAnomaly.get_name(): self._h,
        }

    @property
    def t(self) -> Time:
        """Time."""
        return self._t

    @property
    def u(self) -> ZonalVelocity:
        """Prognostic zonal velocity."""
        return self._u

    @property
    def v(self) -> MeridionalVelocity:
        """Prognostic meriodional velocity."""
        return self._v

    @property
    def h(self) -> LayerDepthAnomaly:
        """Prognostic layer thickness anomaly."""
        return self._h

    @property
    def prognostic(self) -> T:
        """Prognostic variables."""
        return self._prog

    @prognostic.setter
    def prognostic(self, prognostic: T) -> None:
        self._updated()
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    @property
    def vars(self) -> dict[str, PrognosticVariable | BoundDiagnosticVariable]:
        """List of diagnostic variables."""
        return self.prog_vars | self.diag_vars

    @property
    def diag_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Diagnostic variables."""
        return self._diag

    @property
    def prog_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Prognostic variables."""
        return self._prog_vars

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        if not self.diag_vars:
            return [
                "State",
                "└── Prognostic Variables",
                f"\t├── {self.u}",
                f"\t├── {self.v}",
                f"\t└── {self.h}",
            ]
        txt = [
            "State",
            "├── Prognostic Variables",
            f"│\t├── {self.u}",
            f"│\t├── {self.v}",
            f"│\t└── {self.h}",
            "└── Diagnostic Variables",
        ]
        txt_end = [f"\t├── {var}" for var in self.diag_vars.values()]
        chars = txt_end.pop(-1).split()
        chars[0] = "\t└──"
        txt_end.append(" ".join(chars))
        return txt + txt_end

    def __repr__(self) -> str:
        """String representation of State."""
        return "\n".join(self.get_repr_parts())

    def __getitem__(self, name: str) -> BoundDiagnosticVariable:
        """Get bound variables.

        Args:
            name (str): Varibale name

        Raises:
            KeyError: If the name does not correspond to a variable.

        Returns:
            BoundDiagnosticVariable: Bound variable
        """
        if name not in self.vars:
            msg = f"Variables are {', '.join(self.vars.values())}."
            raise KeyError(msg)
        return self.vars[name]

    def _updated(self) -> None:
        """Update diagnostic variables."""
        for var in self.diag_vars.values():
            var.outdated()

    def _time_updated(self) -> None:
        """Update diagnostic variables."""
        for var in filter(lambda v: v.require_time, self.diag_vars.values()):
            var.outdated()

    def add_bound_diagnostic_variable(
        self,
        variable: BoundDiagnosticVariable,
    ) -> None:
        """Add a diagnostic variable.

        Args:
            variable (BoundDiagnosticVariable): Variable.
        """
        if variable.name in self.diag_vars:
            return
        self.diag_vars[variable.name] = variable

    def unbind(self) -> None:
        """Unbind all variables from state."""
        self._diag: dict[str, BoundDiagnosticVariable] = {}

    def update_time(self, time: torch.Tensor) -> None:
        """Update only the value of time.

        Args:
            time (torch.Tensor): Time.
        """
        self._updated()
        prognostic = UVHT.from_uvh(
            time,
            self.prognostic.uvh,
        )
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    def increment_time(self, dt: float) -> None:
        """Increment time."""
        self._time_updated()
        prognostic = self._prog.increment_time(dt)
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    @abstractmethod
    def _update_prognostic_vars(self, prognostic: T) -> None: ...

    @abstractmethod
    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """


class State(BaseState[UVHT]):
    """State: wrapper for UVH state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def _update_prognostic_vars(self, prognostic: UVHT) -> None:
        self._t.update(prognostic.t)
        self._u.update(prognostic.u)
        self._v.update(prognostic.v)
        self._h.update(prognostic.h)

    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """
        self.prognostic = UVHT.from_uvh(self.t.get(), uvh)

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: State.
        """
        return cls(UVHT.steady(n_ens, nl, nx, ny, dtype, device))

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.

        Returns:
            Self: State.
        """
        return cls(UVH(u, v, h))


class StateAlpha(BaseState[UVHTAlpha]):
    """StateAlpha: wrapper for UVHTAlpha state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(self, prognostic: UVHTAlpha) -> None:
        """Instantiate StateAlpha.

        Args:
            prognostic (UVHTAlpha): Core prognostic variables.
        """
        super().__init__(prognostic)
        self._alpha = CollinearityCoefficient(prognostic.alpha)
        self._prog_vars[CollinearityCoefficient.get_name()] = self._alpha

    @property
    def alpha(self) -> CollinearityCoefficient:
        """Collinearity coefficient."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self.update_alpha(alpha)

    def _update_prognostic_vars(self, prognostic: UVHTAlpha) -> None:
        self._t.update(prognostic.t)
        self._u.update(prognostic.u)
        self._v.update(prognostic.v)
        self._h.update(prognostic.h)
        self._alpha.update(prognostic.alpha)

    def _alpha_updated(self) -> None:
        """Update diagnostic variables."""
        for var in filter(lambda v: v.require_alpha, self.diag_vars.values()):
            var.outdated()

    def update_alpha(self, alpha: torch.Tensor) -> None:
        """Update only the value of alpha.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
        """
        self._alpha_updated()
        prognostic = UVHTAlpha.from_uvh(
            self.t.get(),
            alpha,
            self.prognostic.uvh,
        )
        self._prog = prognostic
        self._update_prognostic_vars(prognostic)

    def update_uvh(self, uvh: UVH) -> None:
        """Update u,v and h only.

        Args:
            uvh (UVH): Prognostic u,v and h.
        """
        self.prognostic = UVHTAlpha.from_uvh(
            self.t.get(),
            self.alpha.get(),
            uvh,
        )

    @classmethod
    def steady(
        cls,
        alpha: torch.Tensor,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instantiate a steady state with zero-filled prognostic variables.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: State.
        """
        return cls(UVHTAlpha.steady(alpha, n_ens, nl, nx, ny, dtype, device))

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.
            alpha (torch.Tensor): Collinearity coefficient.

        Returns:
            Self: State.
        """
        return cls(UVHTAlpha(u, v, h, alpha))
