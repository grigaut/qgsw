"""Base classes for variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from qgsw.variables.prognostic import (
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)
from qgsw.variables.uvh import UVH

if TYPE_CHECKING:
    import torch

    from qgsw.variables.base import BoundDiagnosticVariable


class State:
    """State: wrapper for UVH state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(
        self,
        uvh: UVH,
    ) -> None:
        """Instantiate state.

        Args:
            uvh (UVH): Prognostic variables.
        """
        self.unbind()
        self._uvh = uvh
        self._u = ZonalVelocity(uvh.u)
        self._v = MeridionalVelocity(uvh.v)
        self._h = LayerDepthAnomaly(uvh.h)

    @property
    def uvh(self) -> UVH:
        """Prognostic variables."""
        return self._uvh

    @uvh.setter
    def uvh(self, uvh: UVH) -> None:
        self._uvh = uvh
        self._u.update(uvh.u)
        self._v.update(uvh.v)
        self._h.update(uvh.h)
        self._updated()

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
    def diag_vars(self) -> dict[str, BoundDiagnosticVariable]:
        """List of diagnostic variables."""
        return self._diag

    def __getitem__(self, name: str) -> BoundDiagnosticVariable:
        """Get bound variables.

        Args:
            name (str): Varibale name

        Raises:
            KeyError: If the name does not correspond to a variable.

        Returns:
            BoundDiagnosticVariable: Bound variable
        """
        if name not in self.diag_vars:
            msg = f"Bound variables are {', '.join(self.diag_vars.values())}."
            raise KeyError(msg)
        return self.diag_vars[name]

    def update(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Update prognostic variables.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meriodional velocity.
            h (torch.Tensor): Surface height anomaly.
        """
        self.uvh = UVH(u, v, h)

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
        self._diag[variable.name] = variable

    def unbind(self) -> None:
        """Unbind all variables from state."""
        self._diag: dict[str, BoundDiagnosticVariable] = {}

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
