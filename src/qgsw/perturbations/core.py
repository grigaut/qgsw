"""Main perturbation class."""

import torch
from typing_extensions import Self

from qgsw.configs.perturbation import PerturbationConfig
from qgsw.mesh.mesh import Mesh3D
from qgsw.perturbations.base import _Perturbation
from qgsw.perturbations.random import RandomSurfacePerturbation
from qgsw.perturbations.vortex import (
    ActiveLayersRankineVortex3D,
    PassiveLayersRankineVortex3D,
)


class Perturbation:
    """Pertubation class."""

    def __init__(self, perturbation: _Perturbation) -> None:
        """Instantiate the perturbation."""
        self._perturbation = perturbation

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._perturbation.type

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
        return self._perturbation.compute_scale(mesh=mesh)

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
            torch.Tensor: Pressure values, (1, nl, nx, ny)-shaped..
        """
        return self._perturbation.compute_initial_pressure(
            mesh=mesh,
            f0=f0,
            Ro=Ro,
        )

    @classmethod
    def from_config(cls, perturbation_config: PerturbationConfig) -> Self:
        """Instantiate the Perturbation from the configuration.

        Args:
            perturbation_config (PerturbationConfig): Perturbation
            configuration.

        Raises:
            KeyError: If the perturbation is not recognized.

        Returns:
            Self: Perturbation.
        """
        if perturbation_config.type == "vortex-active":
            perturbation = ActiveLayersRankineVortex3D(
                magnitude=perturbation_config.perturbation_magnitude,
            )
            return cls(perturbation)
        if perturbation_config.type == "vortex-passive":
            perturbation = PassiveLayersRankineVortex3D(
                magnitude=perturbation_config.perturbation_magnitude,
            )
            return cls(perturbation)
        if perturbation_config.type == "random-uniform":
            perturbation = RandomSurfacePerturbation(
                magnitude=perturbation_config.perturbation_magnitude,
            )
            return cls(perturbation)
        msg = "Unrecognized perturbation type."
        raise KeyError(msg)