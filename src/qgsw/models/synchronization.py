"""Match models variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qgsw import verbose
from qgsw.exceptions import InvalidLayerNumberError
from qgsw.fields.variables.prognostic_tuples import BasePrognosticUVH
from qgsw.fields.variables.state import BaseStateUVH
from qgsw.models.base import _Model
from qgsw.models.names import ModelCategory, ModelName
from qgsw.models.qg.uvh.core import QGCore
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sw.core import SWCore

ModelRef = TypeVar("ModelRef", bound=_Model)
Model = TypeVar("Model", bound=_Model)
ModelSW = SWCore[BasePrognosticUVH, BaseStateUVH]
ModelQG = QGCore[BasePrognosticUVH, BaseStateUVH, QGProjector]


class ModelSync:
    """Model synchronizing tool."""

    __slots__ = ("_core",)

    def __init__(
        self,
        model_ref: _Model,
        model: _Model,
    ) -> None:
        """Instantiate the model synchronizer.

        Args:
            model_ref (_Model): Reference model.
            model (_Model): Model to synchronize.

        Raises:
            ValueError: If the reference model is a QGPSIQ model.
            ValueError: If the model is a QGPSIQ model.
            ValueError: If model categories don't support synchronization.
        """
        category_ref = model_ref.get_category()
        category = model.get_category()
        sw = ModelCategory.SHALLOW_WATER
        qg = ModelCategory.QUASI_GEOSTROPHIC
        if model_ref.get_type() == ModelName.QUASI_GEOSTROPHIC_USUAL:
            msg = "Synchronization with QGPSIG is not yet possible."
            raise ValueError(msg)
        if model.get_type() == ModelName.QUASI_GEOSTROPHIC_USUAL:
            msg = "Synchronization with QGPSIG is not yet possible."
            raise ValueError(msg)
        if category_ref == sw and category == sw:
            self._core = SWSWSync(model_ref, model)
        elif category_ref == qg and category == qg:
            self._core = QGQGSync(model_ref, model)
        elif category_ref == qg and category == sw:
            self._core = QGSWSync(model_ref, model)
        elif category_ref == sw and category == qg:
            self._core = SWQGSync(model_ref, model)
        else:
            msg = (
                f"Synchronization is not possible between {category_ref}"
                f" and {category} categories."
            )
            raise ValueError(msg)

    def __repr__(self) -> str:
        """String representation of ModelSync."""
        return self._core.__repr__()

    def __call__(self) -> None:
        """Perform synchronization."""
        self._core()


class BaseModelSync(ABC, Generic[ModelRef, Model]):
    """BAse class for model synchronizing."""

    __slots__ = ("_model", "_model_ref", "_nl")

    def __init__(
        self,
        model_ref: ModelRef,
        model: ModelRef,
    ) -> None:
        """Instantiate the BaseModelSync.

        Args:
            model_ref (ModelRef): Reference model.
            model (ModelRef): Model.
        """
        self._raise_if_incompatible_nl(model_ref, model)
        self._model_ref = model_ref
        self._model = model
        self._nl = self._model.space.nl

    def __repr__(self) -> str:
        """String representation of model synchronizer."""
        repr_parts = [
            "Model synchronizer:",
            f"\t├── Reference: {self._model_ref.__class__.__name__}",
            f"\t└── To sync: {self._model.__class__.__name__}",
        ]
        return "/n".join(repr_parts)

    def _raise_if_incompatible_nl(
        self,
        model_ref: ModelRef,
        model: Model,
    ) -> None:
        nl_ref = model_ref.space.nl
        nl = model.space.nl
        if nl_ref >= nl:
            return
        msg = (
            f"Reference model number of layers ({nl_ref}) must be"
            f" greater than model's number of layers ({nl})"
        )
        raise InvalidLayerNumberError(msg)

    @abstractmethod
    def __call__(self) -> None:
        """Perform synchronization."""
        verbose.display("Synchronizing models.", trigger_level=2)


class SWSWSync(BaseModelSync[ModelSW, ModelSW]):
    """Model synchronizer SW->SW."""

    def __call__(self) -> None:
        """Perform synchronization."""
        super().__call__()
        uvh = self._model_ref.prognostic.uvh
        self._model.set_uvh(uvh.parallel_slice[:, : self._nl])


class QGQGSync(BaseModelSync[ModelQG, ModelQG]):
    """Model synchronizer QG->QG."""

    def __call__(self) -> None:
        """Perform synchronization."""
        super().__call__()
        uvh = self._model_ref.prognostic.uvh
        p = self._model_ref.P.compute_p(uvh)[0]
        self._model.set_p(p[:, : self._nl])


class SWQGSync(BaseModelSync[ModelSW, ModelQG]):
    """Model synchronizer SW->QG."""

    def __init__(self, model_ref: ModelSW, model: ModelQG) -> None:
        """Instantiate the BaseModelSync.

        Args:
            model_ref (ModelRef): Reference model.
            model (ModelRef): Model.
        """
        super().__init__(model_ref, model)

    def __call__(self) -> None:
        """Perform synchronization."""
        super().__call__()
        uvh = self._model_ref.prognostic.uvh
        p_qg = self._model_ref.P.compute_p(uvh)[0]
        self._model.set_p(p_qg[:, : self._nl])


class QGSWSync(BaseModelSync[ModelQG, ModelSW]):
    """Model synchronizer QG->SW."""

    def __call__(self) -> None:
        """Perform synchronization."""
        super().__call__()
        nl = self._model.space.nl
        uvh = self._model_ref.prognostic.uvh
        self._model.set_uvh(uvh.parallel_slice[:, :nl])
