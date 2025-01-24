"""Instantiate Model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.qg.core import QG
from qgsw.models.qg.exceptions import UnrecognizedQGModelError
from qgsw.models.qg.modified.collinear_sublayer.core import (
    QGAlpha,
    QGCollinearSF,
)
from qgsw.models.qg.modified.filtered.core import QGCollinearFilteredSF
from qgsw.models.qg.modified.utils import is_modified
from qgsw.models.qg.projector import QGProjector
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.sw.core import SW
from qgsw.models.sw.filtering import (
    SWFilterBarotropicExact,
    SWFilterBarotropicSpectral,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import (
        ModelConfig,
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
    if model_config.type in [
        SW.get_type(),
        SWFilterBarotropicSpectral.get_type(),
        SWFilterBarotropicExact.get_type(),
    ]:
        model = _instantiate_sw(
            model_config=model_config,
            space_2d=space_2d,
            perturbation=perturbation,
            beta_plane=beta_plane,
            Ro=Ro,
        )
    elif model_config.type == QG.get_type():
        model = _instantiate_qg(
            model_config=model_config,
            space_2d=space_2d,
            perturbation=perturbation,
            beta_plane=beta_plane,
            Ro=Ro,
        )
    elif is_modified(model_config.type):
        model = _instantiate_collinear_qg(
            model_config=model_config,
            space_2d=space_2d,
            perturbation=perturbation,
            beta_plane=beta_plane,
            Ro=Ro,
        )
    else:
        msg = f"Unsupported model type: {model_config.type}"
        raise UnrecognizedQGModelError(msg)
    return model


def _instantiate_sw(
    model_config: ModelConfig,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    beta_plane: BetaPlane,
    Ro: float,  # noqa: N803
) -> SW:
    """Instantiate SW model from Configuration.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        SW: SW Model.
    """
    if model_config.type == SW.get_type():
        model_class = SW
    elif model_config.type == SWFilterBarotropicSpectral.get_type():
        model_class = SWFilterBarotropicSpectral
    elif model_config.type == SWFilterBarotropicExact.get_type():
        model_class = SWFilterBarotropicExact
    else:
        msg = f"Unrecognized model type: {model_config.type}"
        raise ValueError(msg)
    model = model_class(
        space_2d=space_2d,
        H=model_config.h,
        g_prime=model_config.g_prime,
        beta_plane=beta_plane,
    )
    p0 = perturbation.compute_initial_pressure(
        model.space.omega,
        beta_plane.f0,
        Ro,
    )
    A = compute_A(  # noqa: N806
        model_config.h,
        model_config.g_prime,
        torch.float64,
        device=DEVICE.get(),
    )
    uvh0 = QGProjector.G(
        p0,
        A=A,
        H=model.H,
        dx=model.space.dx,
        dy=model.space.dy,
        ds=model.space.ds,
        f0=model.beta_plane.f0,
        points_to_surfaces=points_to_surfaces,
    )
    model.set_uvh(
        u=torch.clone(uvh0.u),
        v=torch.clone(uvh0.v),
        h=torch.clone(uvh0.h),
    )
    return model


def _instantiate_qg(
    model_config: ModelConfig,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    beta_plane: BetaPlane,
    Ro: float,  # noqa: N803
) -> QG:
    """Instantiate QG model from Configuration.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        QG: QG Model.
    """
    model = QG(
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
        u=torch.clone(uvh0.u),
        v=torch.clone(uvh0.v),
        h=torch.clone(uvh0.h),
    )
    return model


def _instantiate_collinear_qg(
    model_config: ModelConfig,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    beta_plane: BetaPlane,
    Ro: float,  # noqa: N803
) -> QGAlpha:
    """Instantiate Modified QG Models.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        QGAlpha: Modified QG Model.
    """
    if model_config.type == QGCollinearSF.get_type():
        model_class = QGCollinearSF
    if model_config.type == QGCollinearFilteredSF.get_type():
        model_class = QGCollinearFilteredSF
    else:
        msg = f"Unrecognized model type: {model_config.type}"
        raise ValueError(msg)
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
    model.alpha = _determine_coef0(perturbation.type)
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
        return torch.tensor([0], dtype=torch.float64, device=DEVICE.get())
    if perturbation_type == "vortex-half-barotropic":
        return torch.tensor([0.5], dtype=torch.float64, device=DEVICE.get())
    if perturbation_type == "vortex-barotropic":
        return torch.tensor([1], dtype=torch.float64, device=DEVICE.get())
    if perturbation_type == "none":
        return torch.tensor([1], dtype=torch.float64, device=DEVICE.get())
    msg = f"Unknown perturbation type: {perturbation_type}"
    raise ValueError(msg)
