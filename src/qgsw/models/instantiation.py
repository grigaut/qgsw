"""Instantiate Model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from qgsw.configs.core import Configuration
from qgsw.fields.variables.coefficients.instantiation import instantiate_coef
from qgsw.forcing.wind import WindForcing
from qgsw.models.names import ModelName
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.uvh.core import QG
from qgsw.models.qg.uvh.modified.collinear.core import (
    QGCollinearPV,
    QGCollinearSF,
)
from qgsw.models.qg.uvh.modified.filtered.core import (
    QGCollinearFilteredPV,
    QGCollinearFilteredSF,
)
from qgsw.models.qg.uvh.modified.sanity_check.core import QGSanityCheck
from qgsw.models.qg.uvh.modified.utils import is_modified
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sw.core import SW
from qgsw.models.sw.filtering import (
    SWFilterBarotropicExact,
    SWFilterBarotropicSpectral,
)
from qgsw.models.synchronization.initial_conditions import InitialCondition
from qgsw.perturbations.core import Perturbation
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils import time_params

if TYPE_CHECKING:
    from qgsw.configs.models import (
        ModelConfig,
    )
    from qgsw.configs.perturbation import PerturbationConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.simulations import SimulationConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.configs.windstress import WindStressConfig
    from qgsw.physics.coriolis.beta_plane import BetaPlane

Model = Union[
    SW,
    SWFilterBarotropicSpectral,
    SWFilterBarotropicExact,
    QG,
    QGPSIQ,
    QGCollinearSF,
    QGCollinearFilteredSF,
]


def instantiate_model(
    model_config: ModelConfig,
    beta_plane: BetaPlane,
    space_2d: SpaceDiscretization2D,
    perturbation: Perturbation,
    Ro: float,  # noqa: N803
) -> Model:
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
) -> (
    QG
    | QGCollinearFilteredSF
    | QGCollinearSF
    | QGPSIQ
    | SW
    | SWFilterBarotropicExact
    | SWFilterBarotropicSpectral
):
    """Instantiate SW model from Configuration.

    Args:
        model_config (ModelConfig): Model configuration.
        space_2d (SpaceDiscretization2D): Space Discretization.
        perturbation (Perturbation): Perturbation.
        beta_plane (BetaPlane): Physics configuration.
        Ro (float): Rossby Number.

    Returns:
        QG | QGCollinearFilteredSF | QGCollinearSF | QGPSIQ | SW  : Model.
    """
    model_class = get_model_class(model_config)
    model = model_class(
        space_2d=space_2d,
        H=model_config.h,
        g_prime=model_config.g_prime,
        beta_plane=beta_plane,
    )
    model.name = model_config.name
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
    model.name = model_config.name
    p0 = perturbation.compute_initial_pressure(
        model.space.omega,
        model.beta_plane.f0,
        Ro,
    )
    if model_config.type in [
        ModelName.QG_FILTERED_SF,
        ModelName.QG_FILTERED_PV,
    ]:
        model.P.filter.sigma = model_config.sigma
    uvh0 = QGProjector.G(
        p0,
        A=model.A,
        H=model.H,
        dx=model.space.dx,
        dy=model.space.dy,
        ds=model.space.ds,
        f0=model.beta_plane.f0,
        interpolate=interpolate,
    )
    model.set_uvh(
        u=torch.clone(uvh0.u[:, :1, ...]),
        v=torch.clone(uvh0.v[:, :1, ...]),
        h=torch.clone(uvh0.h[:, :1, ...]),
    )
    return model


ModelClass = Union[
    type[SW],
    type[SWFilterBarotropicExact],
    type[SWFilterBarotropicSpectral],
    type[QG],
    type[QGCollinearFilteredSF],
    type[QGCollinearSF],
    type[QGPSIQ],
    type[QGSanityCheck],
]


def get_model_class(  # noqa: C901, PLR0911
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
    if model_type == ModelName.QG_COLLINEAR_PV:
        return QGCollinearPV
    if model_type == ModelName.QG_FILTERED_SF:
        return QGCollinearFilteredSF
    if model_type == ModelName.QG_FILTERED_PV:
        return QGCollinearFilteredPV
    if model_type == ModelName.QG_SANITY_CHECK:
        return QGSanityCheck
    if model_type == ModelName.SW_FILTER_EXACT:
        return SWFilterBarotropicExact
    if model_type == ModelName.SW_FILTER_SPECTRAL:
        return SWFilterBarotropicSpectral
    msg = f"Unrecognized model type: {model_config.type}"
    raise ValueError(msg)


def instantiate_model_from_config(
    model_config: ModelConfig,
    space: SpaceConfig,
    windstress: WindStressConfig,
    physics: PhysicsConfig,
    perturbation: PerturbationConfig,
    simulation: SimulationConfig,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Model:
    """Instantiate model from configuration.

    Args:
        model_config (ModelConfig): Model configuration.
        space (SpaceConfig): Space configuration.
        windstress (WindStressConfig): Windstress.
        physics (PhysicsConfig): Physics.
        perturbation (PerturbationConfig): Perturbation.
        simulation (SimulationConfig): Simulation.
        dtype (torch.dtype | None, optional): Dtype. Defaults to None.
        device (torch.device | None, optional): Devcie. Defaults to None.

    Returns:
        Model: Model.
    """
    specs = defaults.get(dtype=dtype, device=device)
    # Sub configs

    # Model Set-up
    ## Vortex
    perturbation = Perturbation.from_config(
        perturbation_config=perturbation,
    )
    space_2d = SpaceDiscretization2D.from_config(space)

    model = instantiate_model(
        model_config,
        physics.beta_plane,
        space_2d,
        perturbation,
        Ro=physics.Ro,
    )
    model.slip_coef = physics.slip_coef
    model.bottom_drag_coef = physics.bottom_drag_coefficient
    if is_modified(model.get_type()):
        alpha = instantiate_coef(model_config, space)
        model.alpha = alpha.get()

    if np.isnan(simulation.dt):
        model.dt = time_params.compute_dt(
            model.prognostic.uvh,
            model.space,
            model.g_prime,
            model.H,
        )
    else:
        model.dt = simulation.dt
    model.compute_time_derivatives(model.prognostic.uvh)
    ## Wind Forcing
    wind = WindForcing.from_config(windstress, space, physics)
    taux, tauy = wind.compute()
    model.set_wind_forcing(taux, tauy)
    # Initial condition -------------------------------------------------------
    ic = InitialCondition(model)
    if (startup := simulation.startup) is None:
        ic.set_steady(**specs)
    else:
        ic_conf = Configuration.from_toml(simulation.startup.config)
        ic.set_initial_condition_from_file(
            file=startup.file,
            space_config=ic_conf.space,
            model_config=ic_conf.model,
            physics_config=ic_conf.physics,
            **specs,
        )
    # -------------------------------------------------------------------------
    return model
