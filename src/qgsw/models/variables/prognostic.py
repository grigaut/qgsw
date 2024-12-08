"""Prognostic variables."""

import torch

from qgsw.models.variables.base import PrognosticVariable


class ZonalVelocity(PrognosticVariable[torch.Tensor]):
    """Zonal Velocity."""

    _unit = "s⁻¹"
    _name = "u"
    _description = "Contravariant zonal velocity."


class MeridionalVelocity(PrognosticVariable[torch.Tensor]):
    """Meridional Velocity."""

    _unit = "s⁻¹"
    _name = "v"
    _description = "Contravariant zonal velocity."


class LayerDepthAnomaly(PrognosticVariable[torch.Tensor]):
    """Layer Depth Anomaly."""

    _unit = "m³"
    _name = "h"
    _description = "Layer depth anomaly."
