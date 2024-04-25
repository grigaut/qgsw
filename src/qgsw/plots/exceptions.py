"""Plotting exceptions."""


class InvalidMaskError(Exception):
    """If the mask is invalid."""


class MismatchingMaskError(Exception):
    """Error for when a mask mismatches the dara's shape."""


class AxesInstantiationError(Exception):
    """Error for incorrectly set-up axes."""


class ImpossibleAxesUpdateError(Exception):
    """Error for impossible axis update."""
