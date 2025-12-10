"""Parameters generator for TaylorFullFieldBasis."""

from typing import Any

import torch


def taylor_series(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Generate parameters for taylor time series with full field.

    Args:
        order (int): Order of the time decomposition.
        xx_ref (torch.Tensor): Reference X locations.
        yy_ref (torch.Tensor): Reference Y-locations.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]: Space,
            time.
    """
    xs = xx_ref[:, 0]
    ys = yy_ref[0, :]
    space_params = {
        "xs": xs,
        "ys": ys,
        "numel": xs.numel() * ys.numel(),
    }

    space = {}
    time = {}

    for lvl in range(order):
        space[lvl] = space_params
        time[lvl] = {"numel": 1}

    return space, time
