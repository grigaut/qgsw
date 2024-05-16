"""Base class for perturbations."""

from abc import ABCMeta, abstractmethod

import torch

from qgsw.spatial.core.mesh import Mesh2D, Mesh3D
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

    def __init__(self, magnitude: float = 1e-3) -> None:
        super(TypeSwitch).__init__()
        self._magnitude = magnitude

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._type

    @property
    def magnitude(self) -> float:
        """Perturbation Magnitude."""
        return self._magnitude

    @abstractmethod
    def compute_scale(self, mesh_3d: Mesh3D) -> float:
        """Compute the scale of the perturbation.

        The scale refers to the typical size of the perturbation, not its
        magnitude. For example the scale of a vortex pertubation would be its
        radius.

        Args:
            mesh_3d (Mesh3D): 3D Mesh.

        Returns:
            float: Perturbation scale.
        """

    @abstractmethod
    def _compute_streamfunction_2d(self, mesh_3d: Mesh2D) -> torch.Tensor:
        """Compute the streamfunction for a single layer.

        Args:
            mesh_3d (Mesh2D): Mesh to use for stream function computation.

        Returns:
            torch.Tensor: Stream function values.
        """

    @abstractmethod
    def compute_stream_function(self, mesh_3d: Mesh3D) -> torch.Tensor:
        """Compute the stream function induced by the perturbation.

        Args:
            mesh_3d (Mesh3D): 3D Mesh.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """

    def _adjust_stream_function(
        self,
        psi: torch.Tensor,
        mesh_3d: Mesh3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Adjust stream function values to match Rossby's number.

        Args:
            psi (torch.Tensor): Stream function.
            mesh_3d (Mesh3D): 3D Mesh.
            f0 (float): Coriolis Parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure, (1, nl, nx, ny)-shaped.
        """
        u, v = grad_perp(
            psi,
            mesh_3d.dx,
            mesh_3d.dy,
        )
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        return psi * (Ro * f0 * self.compute_scale(mesh_3d) / u_norm_max)

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
        mesh_3d: Mesh3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Compute the initial pressure values.

        Args:
            mesh_3d (Mesh3D): 3D Mesh.
            f0 (float): Coriolis parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure values.
        """
        psi = self.compute_stream_function(mesh_3d=mesh_3d)
        psi_adjusted = self._adjust_stream_function(psi, mesh_3d, f0, Ro)
        return self._convert_to_pressure(psi=psi_adjusted, f0=f0)


class BarotropicPerturbation(_Perturbation):
    """Barotropic Perturbation."""

    def compute_stream_function(self, mesh_3d: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh_3d (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        psi_2d = self._compute_streamfunction_2d(mesh_3d.remove_z_h())
        nx, ny = psi_2d.shape[-2:]
        return psi_2d.expand((1, mesh_3d.nl, nx, ny))


class BaroclinicPerturbation(_Perturbation):
    """Baroclinic Perturbation."""

    def compute_stream_function(self, mesh_3d: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh_3d (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        psi_2d = self._compute_streamfunction_2d(mesh_3d.remove_z_h())
        nx, ny = psi_2d.shape[-2:]
        psi = torch.ones(
            (1, mesh_3d.nl, nx, ny),
            device=DEVICE,
            dtype=torch.float64,
        )

        psi[0, 0, ...] = psi_2d
        return psi
