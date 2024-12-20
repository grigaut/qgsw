"""Scopes."""

from enum import Enum


class Scope(Enum):
    """Field scope."""

    POINT_WISE = "point-wise"
    LEVEL_WISE = "level-wise"
    ENSEMBLE_WISE = "ensemble-wise"
