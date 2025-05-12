"""Synchronization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qgsw import verbose
from qgsw.models.synchronization.initial_conditions import InitialCondition

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.prognostic_tuples import UVH
    from qgsw.models.base import ModelUVH
    from qgsw.models.names import ModelCategory
    from qgsw.models.qg.uvh.projectors.core import QGProjector


class Synchronizer:
    """Model synchronizer.

    Synchronizes model by using the 'input' model as initial condition
    for the 'output' model.
    """

    __slots__ = (
        "_m_in",
        "_m_out",
        "_model_sync",
        "_nxin",
        "_nxout",
        "_nyin",
        "_nyout",
        "_syncout",
    )

    def __init__(self, model_in: ModelUVH, model_out: ModelUVH) -> None:
        """Instantiate model synchronizer.

        Args:
            model_in (ModelUVH): Model to use as reference.
            model_out (ModelUVH): Model to synchronize with reference.
        """
        # IN specs
        self._m_in = model_in
        self._nxin, self._nyin = model_in.space.nx, model_in.space.ny
        # OUT specs
        self._m_out = model_out
        self._nxout, self._nyout = model_out.space.nx, model_out.space.ny
        self._syncout = InitialCondition(model_out)

    def __call__(self) -> None:
        """Rescale if necessary and Synchronize models."""
        verbose.display(
            msg="Synchronizing model states...",
            trigger_level=1,
        )
        self._syncout.set_initial_condition(
            self._m_in.prognostic.uvh,
            self._m_in.P,
            self._m_in.space.dx,
            self._m_in.space.dy,
            self._m_in.get_category(),
        )
        verbose.display(
            msg=(
                f"'{self._m_out.name}' model state now "
                f"matches '{self._m_in.name}' state."
            ),
            trigger_level=1,
        )

    def sync_to_uvh(
        self,
        uvh: UVH,
        qg_proj: QGProjector,
        *,
        dx: float,
        dy: float,
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
        syncin = InitialCondition(self._m_in)
        syncin.set_initial_condition(
            uvh,
            qg_proj,
            dx,
            dy,
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
        """Set initial condition from file.

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
        syncin = InitialCondition(self._m_in)
        syncin.set_initial_condition_from_file(
            Path(file),
            space_config=space_config,
            model_config=model_config,
            physics_config=physics_config,
            dtype=dtype,
            device=device,
        )
        self()
