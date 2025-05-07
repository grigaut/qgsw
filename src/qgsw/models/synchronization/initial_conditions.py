"""Initial conditions."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from qgsw.exceptions import UnspecifiedConditionCategoryError
from qgsw.fields.variables.prognostic_tuples import (
    UVH,
)
from qgsw.models.base import ModelUVH
from qgsw.models.names import ModelCategory
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.spatial.core.discretization import SpaceDiscretization3D


class InitialCondition:
    """Initial condition."""

    __slots__ = ("_P", "_input_cat", "_model", "_nl")

    def __init__(self, qg_proj_in: QGProjector, model_out: ModelUVH) -> None:
        """Instantiate the initial condition.

        Args:
            qg_proj_in (QGProjector): QG Projector to set initial for model_in.
                This projector is the one associated with the model
                which computed uvh.
            model_out (ModelUVH): Model to set the initial condition for.
        """
        self._raise_if_incompatible_spaces(qg_proj_in.space, model_out.space)
        self._P = qg_proj_in
        self._model = model_out
        self._nl = model_out.space.nl

    @property
    def input_condition_category(self) -> ModelCategory:
        """Input condition model category."""
        try:
            return self._input_cat
        except AttributeError as e:
            msg = "Input condition category not set."
            raise UnspecifiedConditionCategoryError(msg) from e

    @input_condition_category.setter
    def input_condition_category(self, category: ModelCategory) -> None:
        try:
            self._input_cat = ModelCategory(category)
        except ValueError as e:
            msg = f"Unrecognized model category: {category}."
            raise ValueError(msg) from e

    def set_initial_condition(self, uvh: UVH) -> None:
        """Set the initial condition.

        Can only be used if input_condition_category has been set.

        Args:
            uvh (UVH): uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Raises:
            ValueError: If the model category is not recognized.
        """
        if self._input_cat == ModelCategory.QUASI_GEOSTROPHIC:
            return self.set_initial_condition_from_qg(uvh)
        if self._input_cat == ModelCategory.SHALLOW_WATER:
            return self.set_initial_condition_from_sw(uvh)
        msg = f"Unrecognized model category: {self._input_cat}."
        raise ValueError(msg)

    def _raise_if_incompatible_spaces(
        self,
        P_space: SpaceDiscretization3D,  # noqa: N803
        model_space: SpaceDiscretization3D,
    ) -> None:
        """Raise an error if model and projector dimension don't match.

        Args:
            P_space (SpaceDiscretization3D): QG Projector space.
            model_space (SpaceDiscretization3D): Model space.

        Raises:
            ValueError: If model and projector horizontal shape don't match.
        """
        match_nx = P_space.nx == model_space.nx
        match_ny = P_space.ny == model_space.ny

        if not (match_nx and match_ny):
            msg = (
                f"Projector horizontal shape ({P_space.nx}, {P_space.ny}) "
                "must be equal to model horizontal shape nx "
                f"({model_space.nx}, {model_space.ny})"
            )
            raise ValueError(msg)

    def _raise_if_incompatible_uvh_and_qg_proj_in(
        self,
        P_space: SpaceDiscretization3D,  # noqa: N803
        uvh: UVH,
    ) -> None:
        """Raise an error if uvh and projector dimension don't match.

        Args:
            P_space (SpaceDiscretization3D): QG Projector space.
            uvh (UVH): uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Raises:
            ValueError: If nx, ny or nl don't match.
        """
        _, nl, nx, ny = uvh.h.shape

        match_nx = P_space.nx == nx
        match_ny = P_space.ny == ny
        match_nl = P_space.nl == nl

        if not (match_nx and match_ny and match_nl):
            msg = (
                f"Projector shape ({P_space.nl}, {P_space.nx}, {P_space.ny}) "
                f"must be equal to uvh shape nx ({nl}, {nx}, {ny})"
            )
            raise ValueError(msg)

    def set_initial_condition_from_sw(self, uvh: UVH) -> None:
        """Set initial condition from SW uvh.

        Args:
            uvh (UVH): uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        self._raise_if_incompatible_uvh_and_qg_proj_in(self._P.space, uvh)
        if self._model.get_category() == ModelCategory.QUASI_GEOSTROPHIC:
            p_qg = self._P.compute_p(uvh)[0]
            self._model.set_p(p_qg[:, : self._nl])
            return
        self._model.set_uvh(*uvh.parallel_slice[:, : self._nl])

    def set_initial_condition_from_qg(self, uvh: UVH) -> None:
        """Set initial condition from QG uvh.

        Args:
            uvh (UVH): uvh to use as reference: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        self._raise_if_incompatible_uvh_and_qg_proj_in(self._P.space, uvh)
        if self._model.get_category() == ModelCategory.SHALLOW_WATER:
            self._model.set_uvh(*uvh.parallel_slice[:, : self._nl])
            return
        if self._model.space.nl == uvh.h.shape[-3]:
            self._model.set_uvh(*uvh)
            return
        p_qg = self._P.compute_p(uvh)[0]
        self._model.set_p(p_qg[:, : self._nl])

    @classmethod
    def from_models(
        cls,
        model_in: ModelUVH,
        model_out: ModelUVH,
    ) -> Self:
        """Set initial condition from in and out models.

        Args:
            model_in (ModelUVH): Model from which reference uvh comes.
            model_out (ModelUVH): Model to update uvh of.

        Returns:
            Self: Initial condition.
        """
        qg_proj_in = model_in.P.to_shape(
            nx=model_out.space.nx,
            ny=model_out.space.ny,
        )
        return cls(qg_proj_in, model_out)
