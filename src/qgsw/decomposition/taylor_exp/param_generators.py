"""Parameters generators."""

import itertools
from math import log, sqrt
from typing import Any

import torch


def regular_spacing(
    order: int,
    spacing: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Subdivide space into regular-spaced chunks.

    Compute standard deviation for gaussian enveloppes
    that cross at 0.5.

    Args:
        order (int): Taylor series order.
        spacing (int): Width of the chunks.
        xx_ref (torch.Tensor): Reference x locations.
        yy_ref (torch.Tensor): Reference Y locations.

    Returns:
        dict[int,dict[str, Any]]: 0 -> parameters.
    """
    Nx, Ny = xx_ref.shape
    nx = int((Nx) / spacing)
    ny = int((Ny) / spacing)

    offset_x = int((Nx - spacing * nx) / 2)
    offset_y = int((Ny - spacing * ny) / 2)

    xc = xx_ref[offset_x + int(spacing / 2) :: spacing, 0].cpu().flatten()
    yc = yy_ref[0, offset_y + int(spacing / 2) :: spacing].cpu().flatten()

    centers = [(x.item(), y.item()) for x, y in itertools.product(xc, yc)]

    sx = (
        (xx_ref[1, 0] - xx_ref[0, 0]).cpu().item() * spacing / 2 / sqrt(log(2))
    )
    sy = (
        (yy_ref[0, 1] - yy_ref[0, 0]).cpu().item() * spacing / 2 / sqrt(log(2))
    )

    space_params = {
        "centers": centers,
        "sigma_x": sx,
        "sigma_y": sy,
        "numel": len(centers),
    }

    space = {}
    time = {}

    for lvl in range(order):
        space[lvl] = space_params
        time[lvl] = {"numel": 1}

    return space, time
