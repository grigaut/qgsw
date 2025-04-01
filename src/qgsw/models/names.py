"""Model Types."""

from qgsw.utils.named_object import Name


class ModelName(Name):
    """Model names."""

    SHALLOW_WATER = "SW"
    QUASI_GEOSTROPHIC_USUAL = "QGPSIQ"
    QUASI_GEOSTROPHIC = "QG"
    QG_COLLINEAR_SF = "QGCollinearSF"
    QG_FILTERED = "QGCollinearFilteredSF"
    QG_SANITY_CHECK = "QGSanityCheck"
    SW_FILTER_SPECTRAL = "SWFilterBarotropicSpectral"
    SW_FILTER_EXACT = "SWFilterBarotropicExact"
