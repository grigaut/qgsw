"""Model setup."""

from ast import TypeVar

import torch

from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization2D

MODELS = {
    QGPSIQForced.get_type(): QGPSIQForced,
    QGPSIQRGPsi2TransportDR.get_type(): QGPSIQRGPsi2TransportDR,
    QGPSIQForcedSurfML.get_type(): QGPSIQForcedSurfML,
}

M = TypeVar("M", bound=QGPSIQCore)


def set_inhomogeneous_model(
    model: QGPSIQCore, dt: float, slip_coef: float
) -> QGPSIQCore:
    """Set up inhomogeneous model."""
    nx, ny = model.space.nx, model.space.ny
    model.masks = Masks.empty_tensor(nx, ny, device=model.device.get())
    model.bottom_drag_coef = 0
    model.wide = True
    model.slip_coef = slip_coef
    model.dt = dt
    return model


def setup_model(
    model_type: str,
    space: SpaceDiscretization2D,
    H: torch.Tensor,
    g_prime: torch.Tensor,
    beta_plane: BetaPlane,
    dt: float,
    slip_coef: float,
) -> QGPSIQCore:
    """Setup model.

    Args:
        model_type (str): Model type.
        space (SpaceDiscretization2D): Space.
        H (torch.Tensor): H.
        g_prime (torch.Tensor): g_prime.
        beta_plane (BetaPlane): Beta plane.
        dt (float): Timestep.
        slip_coef (float): Slip coefficient.

    Returns:
        QGPSIQCore: Model.
    """
    if model_type == QGPSIQForced.get_type():
        model = _setup_forced(model_type, space, H, g_prime, beta_plane)
    else:
        model = _setup(model_type, space, H, g_prime, beta_plane)
    return set_inhomogeneous_model(model, dt, slip_coef)


def _setup_forced(
    model_type: str,
    space: SpaceDiscretization2D,
    H: torch.Tensor,
    g_prime: torch.Tensor,
    beta_plane: BetaPlane,
) -> QGPSIQForced:
    model = MODELS[model_type](
        space_2d=space,
        H=H[:1] * H[1:2] / (H[:1] + H[1:2]),
        beta_plane=beta_plane,
        g_prime=g_prime[1:2],
    )
    model.wind_scaling = H[:1].item()
    return model


def _setup(
    model_type: str,
    space: SpaceDiscretization2D,
    H: torch.Tensor,
    g_prime: torch.Tensor,
    beta_plane: BetaPlane,
) -> QGPSIQCore:
    return MODELS[model_type](
        space_2d=space,
        H=H[:2],
        beta_plane=beta_plane,
        g_prime=g_prime[:2],
    )
