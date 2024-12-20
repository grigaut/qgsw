"""Utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qgsw.fields.variables.base import Variable


def check_unit_compatibility(*variables: Variable) -> bool:
    """Check if given variables have the same unit as self.

    Args:
        *variables (variable): Variables.

    Returns:
        bool: Whether they have the same unit.
    """
    return all(variables[0].unit == v.unit for v in variables)
