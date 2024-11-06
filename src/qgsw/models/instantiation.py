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
from qgsw.models.qg.core import QG, G, compute_A
from qgsw.models.qg.exceptions import UnrecognizedQGModelError
from qgsw.models.sw.core import SW
from qgsw.models.sw.filtering import (
    SWFilterBarotropicExact,
    SWFilterBarotropicSpectral,
)
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.utils import time_params

if TYPE_CHECKING:
    from qgsw.configs.core import Configuration
    from qgsw.perturbations.core import Perturbation


def instantiate_model(
    config: Configuration,
    space_3d: SpaceDiscretization3D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> (
    SW
    | SWFilterBarotropicSpectral
    | SWFilterBarotropicExact
    | QG
    | QGCollinearPV
    | QGCollinearSF
):
    """Instantiate the model, given the configuration and the perturbation.

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
    match config.model.type:
        case "SW" | "SWFilterBarotropicSpectral" | "SWFilterBarotropicExact":
            model = _instantiate_sw(
                config=config,
                space_3d=space_3d,
                perturbation=perturbation,
                Ro=Ro,
            )
        case "QG":
            model = _instantiate_qg(
                config=config,
                space_3d=space_3d,
                perturbation=perturbation,
                Ro=Ro,
            )
        case "QGCollinearSF" | "QGCollinearPV" | "QGSmoothCollinearSF":
            model = _instantiate_collinear_qg(
                config=config,
                space_3d=space_3d,
                perturbation=perturbation,
                Ro=Ro,
            )
        case _:
            msg = f"Unsupported model type: {config.model.type}"
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


def _instantiate_sw(
    config: Configuration,
    space_3d: SpaceDiscretization3D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> SW:
    match config.model.type:
        case "SW":
            model_class = SW
        case "SWFilterBarotropicSpectral":
            model_class = SWFilterBarotropicSpectral
        case "SWFilterBarotropicExact":
            model_class = SWFilterBarotropicExact
        case _:
            msg = f"Unrecognized model type: {config.model.type}"
            raise ValueError(msg)
    model = model_class(
        space_3d=space_3d,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
    )
    p0 = perturbation.compute_initial_pressure(
        space_3d.omega,
        config.physics.beta_plane.f0,
        Ro,
    )
    A = compute_A(  # noqa: N806
        space_3d.h.xyh.h[:, 0, 0],
        config.model.g_prime.unsqueeze(1).unsqueeze(1)[:, 0, 0],
        torch.float64,
    )
    uvh0 = G(
        p0,
        model.space,
        model.H,
        A,
        model.beta_plane.f0,
    )
    model.set_uvh(
        u=torch.clone(uvh0.u),
        v=torch.clone(uvh0.v),
        h=torch.clone(uvh0.h),
    )
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
    match config.model.type:
        case "QG":
            model_class = QG
        case "QGCollinearSF":
            model_class = QGCollinearSF
        case "QGCollinearPV":
            model_class = QGCollinearPV
        case "QGSmoothCollinearSF":
            model_class = QGSmoothCollinearSF
        case _:
            msg = f"Unrecognized model type: {config.model.type}"
            raise ValueError(msg)
    model = model_class(
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
