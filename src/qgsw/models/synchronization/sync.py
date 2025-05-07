"""Synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.models.synchronization.initial_conditions import InitialCondition
from qgsw.models.synchronization.rescaling import Rescaler

if TYPE_CHECKING:
    from qgsw.fields.variables.prognostic_tuples import UVH
    from qgsw.models.base import ModelUVH
    from qgsw.models.names import ModelCategory
    from qgsw.models.qg.uvh.projectors.core import QGProjector


class Synchronizer:
    """Model synchronizer."""

    __slots__ = (
        "_m_in",
        "_m_out",
        "_model_sync",
        "_nxin",
        "_nxout",
        "_nyin",
        "_nyout",
        "_rescaler",
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
        self._syncout = InitialCondition.from_models(model_in, model_out)
        self._syncout.input_condition_category = self._m_in.get_category()
        # Wether scaling is required or not
        require_scaling = (
            self._nxin != self._nxout or self._nyin != self._nyout
        )
        if require_scaling:
            # Check shapes
            self._raise_if_incompatible_shapes(
                nxin=self._nxin,
                nyin=self._nyin,
                nxout=self._nxout,
                nyout=self._nyout,
            )
            self._rescaler = Rescaler.for_model(self._m_out)
            self._model_sync = self._rescale_sync
        else:
            self._model_sync = self._sync

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

    def _rescale_sync(self) -> None:
        """Rescale then synchronize."""
        # Rescale the input model
        prognostic = self._rescaler(
            self._m_in.prognostic,
            self._m_in.space.dx,
            self._m_in.space.dy,
        )
        # Synchronize the input model to the output model
        self._syncout.set_initial_condition(prognostic.uvh)

    def _sync(self) -> None:
        """Synchronize models."""
        # Sync the input model to the output model
        self._syncout.set_initial_condition(self._m_in.prognostic.uvh)

    def __call__(self) -> None:
        """Rescale if necessary and Synchronize models."""
        self._model_sync()

    def sync_to_uvh(
        self,
        uvh: UVH,
        qg_proj: QGProjector,
        *,
        dx: float,
        dy: float,
        initial_condition_cat: str | ModelCategory,
    ) -> None:
        """Syncrhonize both models to a given uvh.

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
        _, _, nx, ny = uvh.h.shape
        require_scaling = nx != self._nxin or ny != self._nyin
        if require_scaling:
            self._raise_if_incompatible_shapes(
                nxin=nx,
                nyin=ny,
                nxout=self._nxin,
                nyout=self._nyin,
            )
            rescaler = Rescaler.for_model(self._m_in)
            uvh_i = rescaler(uvh, dx, dy)
            qg_proj_in = qg_proj.to_shape(self._nxin, self._nyin)
        else:
            uvh_i = uvh
            qg_proj_in = qg_proj
        syncin = InitialCondition(qg_proj_in, self._m_in)
        syncin.input_condition_category = initial_condition_cat
        syncin.set_initial_condition(uvh_i)
        self()
