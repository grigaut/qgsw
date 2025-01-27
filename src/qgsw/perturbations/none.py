"""Null perturbation."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw.configs.perturbation import PerturbationConfig
from qgsw.perturbations.base import BarotropicPerturbation
from qgsw.spatial.core.grid import Grid2D, Grid3D
from qgsw.specs import DEVICE


class NoPerturbation(BarotropicPerturbation):
    """No Perturbation."""

    _type = "none"

    def __init__(self) -> None:
        """Instantiate NoPerturbation."""

    def compute_scale(self, grid_3d: Grid3D) -> float:  # noqa: ARG002
        """Compute the scale of the perturbation.

        The scale refers to the typical size of the perturbation, not its
        magnitude. For example the scale of a vortex pertubation would be its
        radius.

        Args:
            grid_3d (Grid3D): 3D Grid.

        Returns:
            float: Perturbation scale.
        """
        return 0

    def _compute_streamfunction_2d(self, grid_2d: Grid2D) -> torch.Tensor:
        """Compute the streamfunction for a single layer.

        Args:
            grid_2d (Grid2D): Grid to use for stream function computation.

        Returns:
            torch.Tensor: Stream function filled with 0s.
                └── (nx, ny)-shaped
        """
        return torch.zeros(
            (grid_2d.nx, grid_2d.ny),
            device=DEVICE.get(),
            dtype=torch.float64,
        )

    def compute_initial_pressure(
        self,
        grid_3d: Grid3D,
        f0: float,  # noqa: ARG002
        Ro: float,  # noqa: N803, ARG002
    ) -> torch.Tensor:
        """Compute the initial pressure values.

        Args:
            grid_3d (Grid3D): 3D Grid.
            f0 (float): Coriolis parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure values, filled of 0s.
                └── (1, nl, nx, ny)-shaped
        """
        return self.compute_stream_function(grid_3d)

    @classmethod
    def from_config(cls, perturbation_config: PerturbationConfig) -> Self:  # noqa: ARG003
        """Instantiate Perturbation from config.

        Args:
            perturbation_config (PerturbationConfig): Perturbation Config.

        Returns:
            Self: Perturbation.
        """
        return cls()
