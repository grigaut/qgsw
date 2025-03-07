"""Exceptions."""


class UnsetSigmaError(Exception):
    """When the property sigma is not yet set."""


class UnsetValuesError(Exception):
    """When the property values is not yet set."""


class UnsetLocationsError(Exception):
    """When the property locations is not yet set."""
