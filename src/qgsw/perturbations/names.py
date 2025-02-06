"""Pertubations name."""

from qgsw.utils.named_object import Name


class PertubationName(Name):
    """Pertubation name."""

    BAROTROPIC_VORTEX = "vortex-barotropic"
    HALF_BAROTROPIC_VORTEX = "vortex-half-barotropic"
    BAROCLINIC_VORTEX = "vortex-baroclinic"
    NONE = "none"
