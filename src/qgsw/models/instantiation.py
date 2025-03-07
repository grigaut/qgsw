"""Instantiate Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.names import ModelName
from qgsw.models.qg.projected.core import QG
from qgsw.models.qg.projected.modified.collinear.core import (
    QGCollinearSF,
)
from qgsw.models.qg.projected.modified.filtered.core import (
    QGCollinearFilteredSF,
)
from qgsw.models.qg.projected.modified.sanity_check.core import QGSanityCheck
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.models.qg.projected.projectors.core import QGProjector
from qgsw.models.qg.usual.core import QGPSIQ
from qgsw.models.sw.core import SW
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.configs.models import (
        ModelConfig,
    )
    from qgsw.models.sw.filtering import (
        SWFilterBarotropicExact,
        SWFilterBarotropicSpectral,
    )
    from qgsw.perturbations.core import Perturbation
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )


def instantiate_model(
    model_config: ModelConfig,
    beta_plane: BetaPlane,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> (
    SW
    | SWFilterBarotropicSpectral
    | SWFilterBarotropicExact
    | QG
    | QGPSIQ
    | QGCollinearSF
    | QGCollinearFilteredSF
):
    """Instantiate the model, given the configuration and the perturbation.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Beta Plane.
        Ro (float): Rossby Number.

    Raises:
        UnrecognizedQGModelError: If the model type is not recognized.

    Returns:
        Model: Model.
    """
    if is_modified(model_config.type):
        model = _instantiate_modified(
            model_config=model_config,
            space_2d=space_2d,
            perturbation=perturbation,
            beta_plane=beta_plane,
            Ro=Ro,
        )
    else:
        model = _instantiate_model(
            model_config=model_config,
            space_2d=space_2d,
            perturbation=perturbation,
            beta_plane=beta_plane,
            Ro=Ro,
        )
    return model


def _instantiate_model(
    model_config: ModelConfig,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    beta_plane: BetaPlane,
    Ro: float,  # noqa: N803
) -> QG | QGCollinearFilteredSF | QGCollinearSF | QGPSIQ | SW:
    """Instantiate SW model from Configuration.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        QG | QGCollinearFilteredSF | QGCollinearSF | QGPSIQ | SW: Model.
    """
    model_class = get_model_class(model_config)
    model = model_class(
        space_2d=space_2d,
        H=model_config.h,
        g_prime=model_config.g_prime,
        beta_plane=beta_plane,
    )
    if model.get_type() == ModelName.QG_SANITY_CHECK:
        omega_grid = model.baseline.space.omega
    else:
        omega_grid = model.space.omega

    p0 = perturbation.compute_initial_pressure(
        omega_grid,
        model.beta_plane.f0,
        Ro,
    )
    model.set_p(
        p0,
    )
    return model


def _instantiate_modified(
    model_config: ModelConfig,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    beta_plane: BetaPlane,
    Ro: float,  # noqa: N803
) -> QGCollinearFilteredSF | QGCollinearSF:
    """Instantiate Modified QG Models.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        QGCollinearFilteredSF | QGCollinearSF: Modified QG Model.
    """
    model_class = get_model_class(model_config)
    model = model_class(
        space_2d=space_2d,
        H=model_config.h,
        g_prime=model_config.g_prime,
        beta_plane=beta_plane,
    )
    p0 = perturbation.compute_initial_pressure(
        model.space.omega,
        model.beta_plane.f0,
        Ro,
    )
    if model_config.type == ModelName.QG_FILTERED:
        model.P.filter.sigma = model_config.sigma
    uvh0 = QGProjector.G(
        p0,
        A=model.A,
        H=model.H,
        dx=model.space.dx,
        dy=model.space.dy,
        ds=model.space.ds,
        f0=model.beta_plane.f0,
        points_to_surfaces=points_to_surfaces,
    )
    model.set_uvh(
        u=torch.clone(uvh0.u[:, :1, ...]),
        v=torch.clone(uvh0.v[:, :1, ...]),
        h=torch.clone(uvh0.h[:, :1, ...]),
    )
    return model


ModelClass = type[
    SW | QG | QGCollinearFilteredSF | QGCollinearSF | QGPSIQ | QGSanityCheck
]


def get_model_class(
    model_config: ModelConfig,
) -> ModelClass:
    """Get the model class.

    Args:
        model_config (ModelConfig): Model configuration.

    Raises:
        ValueError: If the model type is not recognized.

    Returns:
        ModelClass: Model class.
    """
    model_type = model_config.type

    if model_type == ModelName.SHALLOW_WATER:
        return SW
    if model_type == ModelName.QUASI_GEOSTROPHIC:
        return QG
    if model_type == ModelName.QUASI_GEOSTROPHIC_USUAL:
        return QGPSIQ
    if model_type == ModelName.QG_COLLINEAR_SF:
        return QGCollinearSF
    if model_type == ModelName.QG_FILTERED:
        return QGCollinearFilteredSF
    if model_type == ModelName.QG_SANITY_CHECK:
        return QGSanityCheck
    msg = f"Unrecognized model type: {model_config.type}"
    raise ValueError(msg)
