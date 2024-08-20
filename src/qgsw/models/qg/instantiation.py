"""Instantiate Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from qgsw.models.qg.alpha import coefficient_from_config
from qgsw.models.qg.collinear_sublayer import (
    QGCollinearPV,
    QGCollinearSF,
    QGSmoothCollinearSF,
    _QGCollinearSublayer,
)
from qgsw.models.qg.core import QG
from qgsw.models.qg.exceptions import UnrecognizedQGModelError
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.utils import time_params

if TYPE_CHECKING:
    from qgsw.configs.core import Configuration
    from qgsw.perturbations.core import Perturbation

collinear_models = {
    "QGCollinearSF": QGCollinearSF,
    "QGCollinearPV": QGCollinearPV,
    "QGSmoothCollinearSF": QGSmoothCollinearSF,
}


def instantiate_model(
    config: Configuration,
    space_3d: SpaceDiscretization3D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> QG | QGCollinearPV | QGCollinearSF:
    """Instantiate the model, given the configuration an dthe perturbation.

    Args:
        config (Configuration): Configuration.
        space_3d (SpaceDiscretization3D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        Ro (float): Rossby Number.

    Raises:
        UnrecognizedQGModelError: If the model type is not recognized.

    Returns:
        Model: Model.
    """
    if config.model.type == "QG":
        model = _instantiate_qg(
            config=config,
            space_3d=space_3d,
            perturbation=perturbation,
            Ro=Ro,
        )
    elif config.model.type in collinear_models:
        model = _instantiate_collinear_qg(
            config=config,
            space_3d=space_3d,
            perturbation=perturbation,
            Ro=Ro,
        )
    else:
        msg = (
            "Unsupported model type, possible values are: "
            f"{collinear_models.keys()} or QG."
        )
        raise UnrecognizedQGModelError(msg)
    model.slip_coef = config.physics.slip_coef
    model.bottom_drag_coef = config.physics.bottom_drag_coef
    if np.isnan(config.simulation.dt):
        model.dt = time_params.compute_dt(
            model.uvh,
            model.space,
            model.g_prime,
            model.H,
        )
    else:
        model.dt = config.simulation.dt
    model.compute_diagnostic_variables(model.uvh)
    model.compute_time_derivatives(model.uvh)
    return model


def _instantiate_qg(
    config: Configuration,
    space_3d: SpaceDiscretization3D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> QG:
    """Instantiate QG model from Configuration.

    Args:
        config (Configuration): Configuration.
        space_3d (SpaceDiscretization3D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        Ro (float): Rossby Number.

    Returns:
        QG: QG Model.
    """
    model = QG(
        space_3d=space_3d,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
    )
    p0 = perturbation.compute_initial_pressure(
        space_3d.omega,
        config.physics.beta_plane.f0,
        Ro,
    )
    uvh0 = model.G(p0)
    model.set_uvh(
        u=torch.clone(uvh0.u),
        v=torch.clone(uvh0.v),
        h=torch.clone(uvh0.h),
    )
    return model


def _instantiate_collinear_qg(
    config: Configuration,
    space_3d: SpaceDiscretization3D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> _QGCollinearSublayer:
    """Instantiate Modified QG Models.

    Args:
        config (Configuration): Configuration.
        space_3d (SpaceDiscretization3D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        Ro (float): Rossby Number.

    Returns:
        _QGCollinearSublayer: Modified QG Model.
    """
    p0 = perturbation.compute_initial_pressure(
        keep_top_layer(space_3d).omega,
        config.physics.beta_plane.f0,
        Ro,
    )
    model: _QGCollinearSublayer = collinear_models[config.model.type](
        space_3d=space_3d,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
        coefficient=_determine_coef0(perturbation.type),
    )
    uvh0 = model.G(p0)
    model.set_uvh(
        u=torch.clone(uvh0.u),
        v=torch.clone(uvh0.v),
        h=torch.clone(uvh0.h),
    )
    model.coefficient = coefficient_from_config(config.model.collinearity_coef)
    return model


def _determine_coef0(perturbation_type: str) -> float:
    """Compute initial coefficient for Modified QG models initialisation.

    Args:
        perturbation_type (str): Perturbation type.

    Raises:
        ValueError: If the perturbation type is not recognized.

    Returns:
        float: Initial coefficient value.
    """
    if perturbation_type == "vortex-baroclinic":
        return 0
    if perturbation_type == "vortex-half-barotropic":
        return 0.5
    if perturbation_type == "vortex-barotropic":
        return 1
    if perturbation_type == "none":
        return 1
    msg = f"Unknown perturbation type: {perturbation_type}"
    raise ValueError(msg)
