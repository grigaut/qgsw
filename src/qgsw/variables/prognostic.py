"""Prognostic variables."""

from qgsw.variables.base import PrognosticVariable


class ZonalVelocity(PrognosticVariable):
    """Zonal Velocity."""

    _unit = "s⁻¹"
    _name = "u"
    _description = "Contravariant zonal velocity."


class MeridionalVelocity(PrognosticVariable):
    """Meridional Velocity."""

    _unit = "s⁻¹"
    _name = "v"
    _description = "Contravariant zonal velocity."


class LayerDepthAnomaly(PrognosticVariable):
    """Layer Depth Anomaly."""

    _unit = "m³"
    _name = "h"
    _description = "Layer depth anomaly."
