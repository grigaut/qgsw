"""Model Types."""

from qgsw.utils.named_object import Name


class ModelName(Name):
    """Model names."""

    SHALLOW_WATER = "SW"
    QUASI_GEOSTROPHIC = "QG"
    QG_COLLINEAR_SF = "QGCollinearSF"
    QG_FILTERED = "QGCollinearFilteredSF"
    SW_FILTER_SPECTRAL = "SWFilterBaroptropicSpectral"
    SW_FILTER_EXACT = "SWFilterBaroptropicExact"
