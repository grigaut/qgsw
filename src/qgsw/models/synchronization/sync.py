"""Synchronization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from qgsw import verbose
from qgsw.models.base import ModelUVH
from qgsw.models.references.base import Reference
from qgsw.models.synchronization.initial_conditions import InitialCondition

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.prognostic_tuples import UVH
    from qgsw.models.names import ModelCategory
    from qgsw.models.qg.uvh.projectors.core import QGProjector

T = TypeVar("T")


class _Synchronizer(Generic[T], ABC):
    __slots__ = ("_ic", "_model", "_ref")

    def __init__(self, reference: T, model: ModelUVH) -> None:
        self._ref = reference
        self._model = model
        self._ic = InitialCondition(self._model)

    @abstractmethod
    def __call__(self) -> None:
        """Rescale if necessary and synchronize model to ref."""


class ModelSynchronizer(_Synchronizer[ModelUVH]):
    """Model synchronizer.

    Synchronizes model by using the 'input' model as initial condition
    for the 'output' model.
    """

    def __call__(self) -> None:
        """Rescale if necessary and synchronize model to ref."""
        verbose.display(
            msg="Synchronizing model states...",
            trigger_level=1,
        )
        self._ic.set_initial_condition(
            self._ref.prognostic.uvh,
            self._ref.P,
            self._ref.get_category(),
        )
        verbose.display(
            msg=(
                f"'{self._model.name}' model state now "
                f"matches '{self._ref.name}' state."
            ),
            trigger_level=1,
        )

    def sync_to_uvh(
        self,
        uvh: UVH,
        qg_proj: QGProjector,
        *,
        initial_condition_cat: str | ModelCategory,
    ) -> None:
        """Synchronize both models to a given uvh.

        Args:
            uvh (UVH): uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            qg_proj (QGProjector): QG Projector to set initial for model_in.
                This projector is the one associated with the model
                which computed uvh.
            dx (float): Infinitesimal distance in the X direction,
                associated with the model which computed uvh.
            dy (float): Infinitesimal distance in the y direction,
                associated with the model which computed uvh.
            initial_condition_cat (str | ModelCategory): Category of the
                initial condition, hence the category of the model
                which computed uvh.
        """
        syncin = InitialCondition(self._ref)
        syncin.set_initial_condition(
            uvh,
            qg_proj,
            initial_condition_cat,
        )
        self()

    def sync_to_file(
        self,
        file: str | Path,
        *,
        space_config: SpaceConfig,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set initial condition from file for both models.

        Args:
            file (str | Path): File to use as UVH input.
            space_config (SpaceConfig): Space configuration
                associated with input.
            model_config (ModelConfig): Model configuration
                associated with input.
            physics_config (PhysicsConfig): Physics configuration
                associated with input.
            dtype (torch.dtype | None, optional): Dtype. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        syncin = InitialCondition(self._ref)
        syncin.set_initial_condition_from_file(
            Path(file),
            space_config=space_config,
            model_config=model_config,
            physics_config=physics_config,
            dtype=dtype,
            device=device,
        )
        self()


class Synchronizer(_Synchronizer[Reference]):
    """Synchronizer."""

    def __call__(self) -> None:
        """Rescale if necessary and synchronize model to ref."""
        self._ref.at_time(self._model.time.item())
        prognostic = self._ref.load()
        self._ic.set_initial_condition(
            prognostic.uvh,
            self._ref.retrieve_P(),
            self._ref.retrieve_category(),
        )
