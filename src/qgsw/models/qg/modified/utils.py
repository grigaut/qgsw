"""Utils for modified models."""

from qgsw.models.qg.modified.collinear_sublayer.core import QGCollinearSF
from qgsw.models.qg.modified.filtered.core import QGCollinearFilteredSF


def is_modified(model_type: str) -> bool:
    """Check if the model is a modified one.

    Args:
        model_type (str): Model type.

    Returns:
        bool: Whether the model is modified or not.
    """
    if model_type == QGCollinearSF.get_type():
        return True
    if model_type == QGCollinearFilteredSF.get_type():  # noqa: SIM103
        return True
    return False
