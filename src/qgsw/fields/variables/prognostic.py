"""Prognostic variables."""

from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import PrognosticVariable
from qgsw.utils.units._units import Unit


class Time(PrognosticVariable):
    """Time."""

    _scope = Scope.POINT_WISE
    _unit = Unit.S
    _name = "t"
    _description = "Time"


class ZonalVelocity(PrognosticVariable):
    """Zonal Velocity."""

    _scope = Scope.POINT_WISE
    _unit = Unit.M2S_1
    _name = "u"
    _description = "Covariant zonal velocity"


class MeridionalVelocity(PrognosticVariable):
    """Meridional Velocity."""

    _scope = Scope.POINT_WISE
    _unit = Unit.M2S_1
    _name = "v"
    _description = "Covariant meriodional velocity"


class LayerDepthAnomaly(PrognosticVariable):
    """Layer Depth Anomaly."""

    _scope = Scope.POINT_WISE
    _unit = Unit.M3
    _name = "h"
    _description = "Layer depth anomaly"


class CollinearityCoefficient(PrognosticVariable):
    """Collinearity coefficient."""

    _scope = Scope.ENSEMBLE_WISE
    _unit = Unit._
    _name = "alpha"
    _description = "Collinearity coefficient"
