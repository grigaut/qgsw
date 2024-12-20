"""Variable sets."""

from __future__ import annotations

from qgsw.fields.errors.point_wise import RMSE


def create_errors_set() -> dict[str, type[RMSE]]:
    """Create error set.

    Returns:
        dict[str, type[RMSE]]: Mapping between error names and type.
    """
    return {
        RMSE.get_name(): RMSE,
    }
