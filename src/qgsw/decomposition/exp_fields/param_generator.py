"""Parameters generators."""

import itertools
from math import log, sqrt
from typing import Any

import torch


def subdivisions(
    xx_ref: torch.Tensor, yy_ref: torch.Tensor
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute space subdivision in 4.

    Args:
        xx_ref (torch.Tensor): Refrecence X locations.
        yy_ref (torch.Tensor): Reference Y locations.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Space params, time params.
    """
    o = 4

    Lx = xx_ref[-1, 0] - xx_ref[0, 0]
    Ly = yy_ref[0, -1] - yy_ref[0, 0]

    lx = Lx / o
    ly = Ly / o
    lt = 20 * 3600 * 24 / o

    xc = [(2 * k + 1) * lx / 2 for k in range(o)]
    yc = [(2 * k + 1) * ly / 2 for k in range(o)]

    centers = [
        (x.cpu().item(), y.cpu().item()) for x, y in itertools.product(xc, yc)
    ]

    tc = [(2 * k + 1) * lt / 2 for k in range(o)]

    sx = lx / 2 / sqrt(log(2))
    sy = ly / 2 / sqrt(log(2))
    st = lt / 2 / sqrt(log(2))

    return {
        "centers": centers,
        "sigma_x": sx,
        "sigma_y": sy,
        "numel": len(centers),
    }, {
        "centers": tc,
        "sigma_t": st,
        "numel": len(tc),
    }
