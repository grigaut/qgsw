"""Main perturbation class."""

import torch
from typing_extensions import Self

from qgsw.configs.perturbation import PerturbationConfig
from qgsw.perturbations.base import _Perturbation
from qgsw.perturbations.random import RandomSurfacePerturbation
from qgsw.perturbations.vortex import (
    BaroclinicVortex,
    BarotropicVortex,
    PerturbedBaroclinicVortex,
    PerturbedBarotropicVortex,
)
from qgsw.spatial.core.grid import Grid3D


class Perturbation:
    """Pertubation class."""

    def __init__(self, perturbation: _Perturbation) -> None:
        """Instantiate the perturbation."""
        self._perturbation = perturbation

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._perturbation.type

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
        return self._perturbation.compute_scale(grid_3d=grid_3d)

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
            torch.Tensor: Pressure values, (1, nl, nx, ny)-shaped..
        """
        return self._perturbation.compute_initial_pressure(
            grid_3d=grid_3d,
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
        perturbations = {
            BaroclinicVortex.get_type(): BaroclinicVortex,
            BarotropicVortex.get_type(): BarotropicVortex,
            RandomSurfacePerturbation.get_type(): RandomSurfacePerturbation,
            PerturbedBaroclinicVortex.get_type(): PerturbedBaroclinicVortex,
            PerturbedBarotropicVortex.get_type(): PerturbedBarotropicVortex,
        }
        if perturbation_config.type not in perturbations:
            msg = (
                "Unrecognized perturbation type. "
                f"Possible values are {perturbations.keys()}"
            )
            raise KeyError(msg)
        perturbation = perturbations[perturbation_config.type](
            magnitude=perturbation_config.perturbation_magnitude,
        )
        return cls(perturbation)
