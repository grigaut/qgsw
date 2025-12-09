"""Parameters generator for TaylorFullFieldBasis."""

from typing import Any


def taylor_series(
    order: int,
    nx: int,
    ny: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Generate parameters for taylor time series with full field.

    Args:
        order (int): Order of the time decomposition.
        nx (int): Number of points in the x-direction.
        ny (int): Number of points in the y-direction.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]: Space,
            time.
    """
    space_params = {
        "nx": nx,
        "ny": ny,
        "numel": nx * ny,
    }

    space = {}
    time = {}

    for lvl in range(order):
        space[lvl] = space_params
        time[lvl] = {"numel": 1}

    return space, time
