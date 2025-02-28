"""Coefficient Names."""

from qgsw.utils.named_object import Name


class CoefficientName(Name):
    """Coefficient names."""

    UNIFORM = "uniform"
    NON_UNIFORM = "non-uniform"
    SMOOOTH_NON_UNIFORM = "smooth-non-uniform"
    LSR_INFERRED_UNIFORM = "lsr-uniform"
