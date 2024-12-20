"""Prognostic variables."""

from qgsw.fields.variables.base import PrognosticVariable
from qgsw.spatial.units._units import Unit


class ZonalVelocity(PrognosticVariable):
    """Zonal Velocity."""

    _unit = Unit.M2S_1
    _name = "u"
    _description = "Covariant zonal velocity"


class MeridionalVelocity(PrognosticVariable):
    """Meridional Velocity."""

    _unit = Unit.M2S_1
    _name = "v"
    _description = "Covariant meriodional velocity"


class LayerDepthAnomaly(PrognosticVariable):
    """Layer Depth Anomaly."""

    _unit = Unit.M3
    _name = "h"
    _description = "Layer depth anomaly"
