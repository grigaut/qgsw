"""Main module for references."""

from qgsw.configs.core import Configuration
from qgsw.models.references.base import Reference
from qgsw.models.references.model import ModelOutputReference, ModelReference
from qgsw.models.references.names import ReferenceName


def load_reference(configuration: Configuration) -> Reference:
    """Load the reference.

    Args:
        configuration (Configuration): Configuration.

    Returns:
        Reference: Reference.
    """
    ref_type = configuration.simulation.reference.type
    if ref_type == ReferenceName.MODEL:
        return ModelReference.from_config(configuration)
    if ref_type == ReferenceName.MODEL_OUTPUT:
        return ModelOutputReference.from_config(configuration)
    msg = f"Unrecognized Reference type: {ref_type}"
    raise ValueError(msg)
