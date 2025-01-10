"""Base classes for variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)
from qgsw.fields.variables.uvh import UVH, UVHalpha

if TYPE_CHECKING:
    import torch

    from qgsw.fields.variables.base import (
        BoundDiagnosticVariable,
        PrognosticVariable,
    )


T = TypeVar("T", bound=UVH | UVHalpha)


class BaseState(ABC, Generic[T]):
    """Base State class."""

    def __init__(self, uvh: T) -> None:
        """Instantiate State.

        Args:
            uvh (T): Core UVH or UVHalpha.
        """
        self.unbind()
        self._uvh = uvh
        self._u = ZonalVelocity(uvh.u)
        self._v = MeridionalVelocity(uvh.v)
        self._h = LayerDepthAnomaly(uvh.h)
        self._prog = {
            ZonalVelocity.get_name(): self._u,
            MeridionalVelocity.get_name(): self._v,
            LayerDepthAnomaly.get_name(): self._h,
        }

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
    def uvh(self) -> T:
        """Prognostic variables."""
        return self._uvh

    @uvh.setter
    def uvh(self, uvh: T) -> None:
        self._updated()
        self._uvh = uvh
        self._update_prognostic_vars(uvh)

    @property
    def vars(self) -> dict[str, PrognosticVariable | BoundDiagnosticVariable]:
        """List of diagnostic variables."""
        return self._prog | self._diag

    @property
    def diag_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Diagnostic variables."""
        return self._diag

    @property
    def prog_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """Prognostic variables."""
        return self._prog

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

    @abstractmethod
    def _update_prognostic_vars(self, uvh: T) -> None: ...


class State(BaseState[UVH]):
    """State: wrapper for UVH state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def _update_prognostic_vars(self, uvh: UVH) -> None:
        self._u.update(uvh.u)
        self._v.update(uvh.v)
        self._h.update(uvh.h)

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
        return cls(UVH.steady(n_ens, nl, nx, ny, dtype, device))

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


class Statealpha(BaseState[UVHalpha]):
    """Statealpha: wrapper for UVHalpha state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(self, uvh: UVHalpha) -> None:
        """Instantiate Statealpha.

        Args:
            uvh (UVHalpha): Core prognostic variables.
        """
        super().__init__(uvh)
        self._alpha = CollinearityCoefficient(uvh.alpha)
        self._prog[CollinearityCoefficient.get_name()] = self._alpha

    @property
    def alpha(self) -> CollinearityCoefficient:
        """Collinearity coefficient."""
        return self._alpha

    def _update_prognostic_vars(self, uvh: UVHalpha) -> None:
        self._u.update(uvh.u)
        self._v.update(uvh.v)
        self._h.update(uvh.h)
        self._alpha.update(uvh.alpha)

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
        return cls(UVHalpha.steady(alpha, n_ens, nl, nx, ny, dtype, device))

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
        return cls(UVHalpha(u, v, h, alpha))
