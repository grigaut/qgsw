"""Base class for perturbations."""

from abc import ABCMeta, abstractmethod

import torch

from qgsw.spatial.core.grid import Grid2D, Grid3D
from qgsw.specs import DEVICE
from qgsw.utils.type_switch import TypeSwitch


def grad_perp(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Orthogonal gradient."""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


class _Perturbation(TypeSwitch, metaclass=ABCMeta):
    """Perturbation base class."""

    _type: str
    _ratio_sub_top: float

    def __init__(self, magnitude: float = 1e-3) -> None:
        super(TypeSwitch).__init__()
        self._magnitude = magnitude

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._type

    @property
    def layer_ratio(self) -> float:
        """Ratio between sublayers streamfunctions and top layer one."""
        return self._ratio_sub_top

    @property
    def magnitude(self) -> float:
        """Perturbation Magnitude."""
        return self._magnitude

    @abstractmethod
    def compute_scale(self, grid_3d: Grid3D) -> float:
        """Compute the scale of the perturbation.

        The scale refers to the typical size of the perturbation, not its
        magnitude. For example the scale of a vortex pertubation would be its
        radius.

        Args:
            grid_3d (Grid3D): 3D Grid.

        Returns:
            float: Perturbation scale.
        """

    @abstractmethod
    def _compute_streamfunction_2d(self, grid_3d: Grid2D) -> torch.Tensor:
        """Compute the streamfunction for a single layer.

        Args:
            grid_3d (Grid2D): Grid to use for stream function computation.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny) shaped.
        """

    def compute_stream_function(self, grid_3d: Grid3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            grid_3d (Grid3D): 3D Grid to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        psi = torch.ones(
            (1, grid_3d.nl, grid_3d.nx, grid_3d.ny),
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        psi_2d = self._compute_streamfunction_2d(grid_3d.remove_z_h())
        psi[0, 0, ...] = psi_2d
        for i in range(1, grid_3d.nl):
            psi_2d = self._compute_streamfunction_2d(grid_3d.remove_z_h())
            psi[0, i, ...] = self.layer_ratio * psi_2d
        return psi

    def _adjust_stream_function(
        self,
        psi: torch.Tensor,
        grid_3d: Grid3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Adjust stream function values to match Rossby's number.

        Args:
            psi (torch.Tensor): Stream function.
            grid_3d (Grid3D): 3D Grid.
            f0 (float): Coriolis Parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure, (1, nl, nx, ny)-shaped.
        """
        u, v = grad_perp(
            psi,
            grid_3d.dx,
            grid_3d.dy,
        )
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        return psi * (Ro * f0 * self.compute_scale(grid_3d) / u_norm_max)

    def _convert_to_pressure(
        self,
        psi: torch.Tensor,
        f0: float,
    ) -> torch.Tensor:
        """Convert stream function to pressure.

        Args:
            psi (torch.Tensor): Stream function.
            f0 (float): Coriolis Parameter.

        Returns:
            torch.Tensor: Pressure, (1, nl, nx, ny)-shaped.
        """
        return psi * f0

    def compute_initial_pressure(
        self,
        grid_3d: Grid3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Compute the initial pressure values.

        Args:
            grid_3d (Grid3D): 3D Grid.
            f0 (float): Coriolis parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure values.
        """
        psi = self.compute_stream_function(grid_3d=grid_3d)
        psi_adjusted = self._adjust_stream_function(psi, grid_3d, f0, Ro)
        return self._convert_to_pressure(psi=psi_adjusted, f0=f0)


class BarotropicPerturbation(_Perturbation):
    """Barotropic Perturbation."""

    _ratio_sub_top = 1


class BaroclinicPerturbation(_Perturbation):
    """Baroclinic Perturbation."""

    _ratio_sub_top = 0


class HalfBarotropicPerturbation(_Perturbation):
    """Half Barotropic Perturbation."""

    _ratio_sub_top = 0.5
