"""Utils for modified models."""

from qgsw.models.names import ModelName


def is_modified(model_type: str) -> bool:
    """Check if the model is a modified one.

    Args:
        model_type (str): Model type.

    Returns:
        bool: Whether the model is modified or not.
    """
    if model_type == ModelName.QG_COLLINEAR_SF:
        return True
    if model_type == ModelName.QG_FILTERED:  # noqa: SIM103
        return True
    return False
