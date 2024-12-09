"""Variables."""

from qgsw.variables.dynamics import (
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalVorticity,
    PhysicalZonalVelocity,
    PotentialVorticity,
    Pressure,
    SurfaceHeightAnomaly,
    Vorticity,
    ZonalVelocityFlux,
)
from qgsw.variables.energetics import (
    KineticEnergy,
    TotalKineticEnergy,
    TotalModalAvailablePotentialEnergy,
    TotalModalEnergy,
    TotalModalKineticEnergy,
)
from qgsw.variables.prognostic import (
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)
from qgsw.variables.state import State
from qgsw.variables.uvh import UVH

__all__ = [
    "LayerDepthAnomaly",
    "MeridionalVelocity",
    "MeridionalVelocityFlux",
    "PhysicalLayerDepthAnomaly",
    "PhysicalMeridionalVelocity",
    "PhysicalVorticity",
    "PhysicalZonalVelocity",
    "PotentialVorticity",
    "Pressure",
    "KineticEnergy",
    "State",
    "SurfaceHeightAnomaly",
    "TotalKineticEnergy",
    "TotalModalAvailablePotentialEnergy",
    "TotalModalEnergy",
    "TotalModalKineticEnergy",
    "UVH",
    "ZonalVelocity",
    "ZonalVelocityFlux",
    "Vorticity",
]
