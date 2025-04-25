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


class ModelCategory(Name):
    """Model category."""

    SHALLOW_WATER = "Shallow-Water"
    QUASI_GEOSTROPHIC = "Quasi-Geostrophic"


def get_category(model_name: ModelName) -> ModelCategory:  # noqa: PLR0911
    """Get model category from model name.

    Args:
        model_name (ModelName): Model name.

    Raises:
        ValueError: If the model name is not known.

    Returns:
        ModelCategory: Model category.
    """
    if model_name == ModelName.SHALLOW_WATER:
        return ModelCategory.SHALLOW_WATER
    if model_name == ModelName.SW_FILTER_EXACT:
        return ModelCategory.SHALLOW_WATER
    if model_name == ModelName.SW_FILTER_SPECTRAL:
        return ModelCategory.SHALLOW_WATER
    if model_name == ModelName.QUASI_GEOSTROPHIC:
        return ModelCategory.QUASI_GEOSTROPHIC
    if model_name == ModelName.QUASI_GEOSTROPHIC_USUAL:
        return ModelCategory.QUASI_GEOSTROPHIC
    if model_name == ModelName.QG_COLLINEAR_SF:
        return ModelCategory.QUASI_GEOSTROPHIC
    if model_name == ModelName.QG_FILTERED:
        return ModelCategory.QUASI_GEOSTROPHIC
    if model_name == ModelName.QG_SANITY_CHECK:
        return ModelCategory.QUASI_GEOSTROPHIC
    msg = f"Unrecognized model name: {model_name}."
    raise ValueError(msg)
