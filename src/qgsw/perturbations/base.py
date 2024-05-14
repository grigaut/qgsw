"""Base class for perturbations."""

from abc import ABCMeta, abstractmethod

import torch

from qgsw.mesh.mesh import Mesh3D


def grad_perp(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Orthogonal gradient."""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


class _Perturbation(metaclass=ABCMeta):
    """Perturbation base class."""

    _type: str

    def __init__(self, magnitude: float = 1e-3) -> None:
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
    def compute_scale(self, mesh: Mesh3D) -> float:
        """Compute the scale of the perturbation.

        The scale refers to the typical size of the perturbation, not its
        magnitude. For example the scale of a vortex pertubation would be its
        radius.

        Args:
            mesh (Mesh3D): 3D Mesh.

        Returns:
            float: Perturbation scale.
        """

    @abstractmethod
    def compute_stream_function(self, mesh: Mesh3D) -> torch.Tensor:
        """Compute the stream function induced by the perturbation.

        Args:
            mesh (Mesh3D): 3D Mesh.

        Returns:
            torch.Tensor: Stream function values.
        """

    def _convert_to_pressure(
        self,
        psi: torch.Tensor,
        mesh: Mesh3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Convert stream function to pressure.

        Args:
            psi (torch.Tensor): Stream function.
            mesh (Mesh3D): 3D Mesh.
            f0 (float): Coriolis Parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure
        """
        u, v = grad_perp(
            psi,
            mesh.dx,
            mesh.dy,
        )
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        psi_norm = psi * (Ro * f0 * self.compute_scale(mesh) / u_norm_max)
        # Return pressure value
        return psi_norm * f0

    def compute_initial_pressure(
        self,
        mesh: Mesh3D,
        f0: float,
        Ro: float,  # noqa: N803
    ) -> torch.Tensor:
        """Compute the initial pressure values.

        Args:
            mesh (Mesh3D): 3D Mesh.
            f0 (float): Coriolis parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure values.
        """
        psi = self.compute_stream_function(mesh=mesh)
        return self._convert_to_pressure(
            psi=psi,
            mesh=mesh,
            f0=f0,
            Ro=Ro,
        )
