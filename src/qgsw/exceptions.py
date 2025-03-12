"""Exceptions."""


class UnsetSigmaError(Exception):
    """When the property sigma is not yet set."""


class UnsetValuesError(Exception):
    """When the property values is not yet set."""


class UnsetCentersError(Exception):
    """When the property centers is not yet set."""


class InappropriateShapeError(Exception):
    """When a tensor has an inappropriate shape."""
