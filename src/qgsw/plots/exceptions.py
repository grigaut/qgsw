"""Plotting exceptions."""


class InvalidMaskError(Exception):
    """If the mask is invalid."""


class MismatchingMaskError(Exception):
    """Eroor for when a mask mismatches the dara's shape."""
