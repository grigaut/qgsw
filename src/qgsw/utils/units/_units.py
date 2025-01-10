"""Units."""

from enum import Enum


class Unit(Enum):
    """Unit."""

    _ = " "
    S = "s"
    M = "m"
    KM = "km"
    RAD = "rad"
    DEG = "deg"
    S_1 = "s⁻¹"
    S_2 = "s⁻²"
    M3 = "m³"
    M1S_1 = "m.s⁻¹"
    M2S_1 = "m².s⁻¹"
    M2S_2 = "m².s⁻²"

    def __repr__(self) -> str:
        """String representation of unit."""
        return self.value
