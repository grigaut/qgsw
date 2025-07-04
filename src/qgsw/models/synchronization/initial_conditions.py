"""Initial conditions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qgsw import verbose
from qgsw.fields.variables.tuples import (
    UVH,
)
from qgsw.models.names import ModelCategory, get_category
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.synchronization.rescaling import Rescaler
from qgsw.specs import defaults
from qgsw.utils import covphys

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.models.base import ModelUVH


class InitialCondition:
    """Initial condition.

    Perform rescaling using Rescaler if input data shape does not match output
    model shapes.
    """

    __slots__ = ("_input_cat", "_model", "_nl", "_rescaler")

    def __init__(self, model_out: ModelUVH) -> None:
        """Instantiate the initial condition.

        Args:
            qg_proj_in (QGProjector): QG Projector to set initial for model_in.
                This projector is the one associated with the model
                which computed uvh.
            model_out (ModelUVH): Model to set the initial condition for.
        """
        self._model = model_out
        self._rescaler = Rescaler.for_model(self._model)
        self._nl = model_out.space.nl

    def _raise_if_incompatible_shapes(
        self,
        *,
        nxin: int,
        nyin: int,
        nxout: int,
        nyout: int,
    ) -> None:
        """Raise an error if the model have incompatible shapes.

        Args:
            nxin (int): Input nx.
            nyin (int): Input ny.
            nxout (int): Output nx.
            nyout (int): Output ny.

        Raises:
            ValueError: If nxout (or, resp. nxin) does not divide nxin
                (or, resp. nxout).
            ValueError: If nyout (or, resp. nyin) does not divide nyin
                (or, resp. nyout).
            ValueError: If nxout / nxin != nyout / nyin
        """
        if (nxin % nxout != 0) and (nxout % nxin != 0):
            msg = (
                "nxin (or, resp. nxout) must divisable"
                " by nxout (or, resp. nxin)."
            )
            raise ValueError(msg)
        if (nyin % nyout != 0) and (nyout % nyin != 0):
            msg = (
                "nyin (or, resp. nyout) must divisable"
                " by nyout (or, resp. nyin)."
            )
            raise ValueError(msg)
        if nxout / nxin != nyout / nyin:
            msg = (
                "There should be the same ratio between"
                " nxout / nxin and nyout / nyin"
            )
            raise ValueError(msg)

    def _raise_if_different_steps(
        self,
        *,
        dxin: int,
        dyin: int,
        dxout: int,
        dyout: int,
    ) -> None:
        """Raise an error if the model have incompatible shapes.

        Args:
            dxin (int): Input dx.
            dyin (int): Input dy.
            dxout (int): Output dx.
            dyout (int): Output dy.

        Raises:
            ValueError: If dxout does equal dxin.
            ValueError: If dyout does equal dyin.
        """
        if dxin != dxout:
            msg = "dxin must equal dxout for same-sized uvh/models."
            raise ValueError(msg)
        if dyin != dyout:
            msg = "dyin must equal dyout for same-sized uvh/models."
            raise ValueError(msg)

    def _rescale(
        self,
        uvh: UVH,
        proj: QGProjector,
    ) -> tuple[UVH, QGProjector]:
        """Rescale UVH.

        Args:
            uvh (UVH): Input (physical) UVH.
            proj (QGProjector | None, optional): QG projector related with
                input data. If None, the rescaled data won't be QG.
                Defaults to None.

        Returns:
            tuple[UVH, QGProjector]: Interpolated (physical) UVH, QG projector
                whose shapes matches new UVH shape
        """
        _, _, nx, ny = uvh.h.shape
        nx_model, ny_model = self._model.space.nx, self._model.space.ny
        require_scaling = nx != nx_model or ny != ny_model
        if not require_scaling:
            return uvh, proj
        self._raise_if_incompatible_shapes(
            nxin=nx,
            nyin=ny,
            nxout=nx_model,
            nyout=ny_model,
        )
        uvh_i = self._rescaler(uvh, proj)
        proj_i = proj.to_shape(nx_model, ny_model)
        return uvh_i, proj_i

    def set_initial_condition(
        self,
        uvh: UVH,
        proj: QGProjector,
        input_category: str | ModelCategory = ModelCategory.SHALLOW_WATER,
    ) -> None:
        """Set the initial condition.

        Can only be used if input_condition_category has been set.

        Args:
            uvh (UVH): (physical) uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            proj (QGProjector): QGProjector associated with input uvh.
            input_category (str | ModelCategory, optional): Input model
                category. Defaults to ModelCategory.SHALLOW_WATER

        Raises:
            ValueError: If the model category is not recognized.
        """
        if input_category == ModelCategory.QUASI_GEOSTROPHIC:
            return self.set_initial_condition_from_qg(uvh, proj)
        if input_category == ModelCategory.SHALLOW_WATER:
            return self.set_initial_condition_from_sw(uvh, proj)
        msg = f"Unrecognized model category: {input_category}."
        raise ValueError(msg)

    def set_initial_condition_from_sw(
        self,
        uvh: UVH,
        proj: QGProjector,
    ) -> None:
        """Set initial condition from SW uvh.

        Args:
            uvh (UVH): (physical) uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            proj (QGProjector): QGProjector associated with input uvh.
        """
        verbose.display(
            msg=f"Setting initial condition for '{self._model.name}' model.",
            trigger_level=2,
        )
        if self._model.get_category() == ModelCategory.QUASI_GEOSTROPHIC:
            # SW -> QG
            uvh_i, proj_i = self._rescale(uvh, proj)
            uvh_cov = covphys.to_cov(
                uvh_i,
                self._model.space.dx,
                self._model.space.dy,
            )
            p_qg = proj_i.compute_p(uvh_cov)[0]
            self._model.set_p(p_qg)
        else:
            # SW -> SW
            uvh_i, proj_i = self._rescale(uvh, None)
            self._model.set_physical_uvh(*uvh_i.parallel_slice[:, : self._nl])
        verbose.display(
            msg="Initial condition set.",
            trigger_level=2,
        )

    def set_initial_condition_from_qg(
        self,
        uvh: UVH,
        proj: QGProjector,
    ) -> None:
        """Set initial condition from QG uvh.

        Args:
            uvh (UVH): (physical) uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            proj (QGProjector): QGProjector associated with input uvh.
        """
        verbose.display(
            msg=f"Setting initial condition for '{self._model.name}' model.",
            trigger_level=2,
        )
        if self._model.get_category() == ModelCategory.SHALLOW_WATER:
            # QG -> SW
            uvh_i, proj_i = self._rescale(uvh, None)
            self._model.set_physical_uvh(*uvh_i.parallel_slice[:, : self._nl])
        elif self._model.space.nl == uvh.h.shape[-3]:
            # QG -> QG with same number of layers
            uvh_i, _ = self._rescale(uvh, proj)
            self._model.set_physical_uvh(*uvh_i)
        else:
            # QG -> QG
            uvh_i, proj_i = self._rescale(uvh, proj)
            uvh_cov = covphys.to_cov(
                uvh_i,
                self._model.space.dx,
                self._model.space.dy,
            )
            p_qg = proj_i.compute_p(uvh_cov)[0]
            self._model.set_p(p_qg)
        verbose.display(
            msg="Initial condition set.",
            trigger_level=2,
        )

    def set_initial_condition_from_file(
        self,
        file: str | Path,
        *,
        space_config: SpaceConfig = None,
        model_config: ModelConfig = None,
        physics_config: PhysicsConfig = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set initial condition from file.

        Args:
            file (str | Path): File to use as (physical) UVH input.
            space_config (SpaceConfig): Space configuration
                associated with input.
            model_config (ModelConfig): Model configuration
                associated with input.
            physics_config (PhysicsConfig): Physics configuration
                associated with input.
            dtype (torch.dtype | None, optional): Dtype. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        specs = defaults.get(dtype=dtype, device=device)
        file = Path(file)
        self.set_initial_condition(
            UVH.from_file(
                Path(file),
                **specs,
            ),
            proj=QGProjector.from_config(
                space_config,
                model_config,
                physics_config,
                **specs,
            ),
            input_category=get_category(model_config.type),
        )

    def set_steady(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set steady (physical) uvh.

        Args:
            dtype (torch.dtype | None, optional): Dtype. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        verbose.display(
            msg=(
                f"Setting steady initial condition for '{self._model.name}' "
                "model."
            ),
            trigger_level=2,
        )
        specs = defaults.get(dtype=dtype, device=device)
        n_ens = 1
        nl = self._model.space.nl
        nx = self._model.space.nx
        ny = self._model.space.ny
        uvh = UVH.steady(
            n_ens=n_ens,
            nl=nl,
            nx=nx,
            ny=ny,
            **specs,
        )
        self._model.set_physical_uvh(*uvh)
        verbose.display(
            msg="Initial condition set.",
            trigger_level=2,
        )
