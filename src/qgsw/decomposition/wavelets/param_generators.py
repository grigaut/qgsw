"""Wavemet parameters generation."""

from __future__ import annotations

import itertools
from math import log, sqrt
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from qgsw import specs
from qgsw.logging.core import getLogger
from qgsw.logging.utils import meters2text, sec2text, step

logger = getLogger(__name__)


def dyadic_decomposition(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
    sigma_t_sigma_xy_ratio: float = 1.17,
    sigma_xy_l_p_ratio: float = 1.75 / torch.pi * 2 * sqrt(2),
    l_xy_sigma_xy_ratio: float = 2 * sqrt(log(2)),
    l_t_sigma_t_ratio: float = 2 * sqrt(log(2)),
    use_time_frontier_as_centers: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Compute wavelet parameters using dyadic subdivisions.

    Args:
        order (int): Decomposition order (number of subdivision to perform).
        xx_ref (torch.Tensor): Reference x locations to use to place centers.
        yy_ref (torch.Tensor): Reference y locations to use to place centers.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.
        sigma_t_sigma_xy_ratio (float, optional): Ratio between time gaussian
            enveloppe and space gaussian enveloppe. Default value is set so
            to roughly match 900 km to 20 days and 80 km to 2 days.
            Defaults to 2.
        sigma_xy_l_p_ratio (float, optional): Ratio between space gaussian
            enveloppe std and subdivide length. Defaults set so that σ' allows
            exp((-x²-y²)/σ') ~ cos²(πx/(2σ))cos²(πy/(2σ)) close to 0.
            Defaults to 1.75/π*2*√2.
        l_xy_sigma_xy_ratio (float, optional): Ratio between space patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        l_t_sigma_t_ratio (float, optional): Ratio between time patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        use_time_frontier_as_centers (bool, optional): Center gaussian
            enveloppes on time boundaries. If set to True, no guarante that
            time enveloppes will overlap at 0.5. Defaults to False.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    Lx = (xx_ref[-1, 0] - xx_ref[0, 0]).item()
    x0 = xx_ref[0, 0].item()
    Ly = (yy_ref[0, -1] - yy_ref[0, 0]).item()
    y0 = yy_ref[0, 0].item()

    space = {}
    time = {}

    with logger.section(f"Performing {order} dyadic subdivisions..."):
        for p in range(order):
            # Subdivide Lxy_max
            l_p = Lxy_max / 2**p
            # Compute k
            k = 2 * torch.pi / l_p
            # Compute associated std for spatial gaussian enveloppe
            sigma_xy = l_p * sigma_xy_l_p_ratio
            # Compute space patches width
            l_xy = l_xy_sigma_xy_ratio * sigma_xy
            # Compute number of patches requires in both direction
            nx = int(Lx / l_xy) + 1
            ny = int(Ly / l_xy) + 1
            # Compute required offset to center patches on the domain
            offset_x = Lx - (l_xy * (nx - 1))
            offset_y = Ly - (l_xy * (ny - 1))
            # Compute patch centers
            xc = [x0 + offset_x * 0.5 + i * l_xy for i in range(nx)]
            yc = [y0 + offset_y * 0.5 + i * l_xy for i in range(ny)]

            xyc = [(x, y) for x, y in itertools.product(xc, yc)]

            space[p] = {
                "centers": xyc,
                "kx": k,
                "ky": k,
                "sigma_x": sigma_xy,
                "sigma_y": sigma_xy,
                "numel": len(xyc),
            }
            # Compute std for time gaussian enveloppe
            sigma_t = sigma_t_sigma_xy_ratio * sigma_xy
            # Compute associated time patch size
            l_t = l_t_sigma_t_ratio * sigma_t
            # Compute required number of time patches
            nt = int(Lt_max // l_t) + 1

            if use_time_frontier_as_centers:
                # Compute adjusted patch lengths
                l_t = Lt_max / nt
                # Compute patch centers
                tc = [i * l_t for i in range(nt + 1)]
            else:
                # Compute required offset to center patches on the domain
                offset_t = Lt_max - (nt - 1) * l_t
                # Compute patch centers
                tc = [offset_t * 0.5 + i * l_t for i in range(nt)]

            time[p] = {
                "centers": tc,
                "sigma_t": sigma_t,
                "numel": len(tc),
            }
            msg = (
                f"[Subdivision {step(p + 1, order)}] {meters2text(l_xy)} x "
                f"{meters2text(l_xy)} space patches associated to"
                f" {sec2text(l_t)} long time patches."
            )
            logger.detail(msg)

    return space, time


def linear_decomposition(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
    sigma_t_sigma_xy_ratio: float = 1.17,
    sigma_xy_l_p_ratio: float = 1.75 / torch.pi * 2 * sqrt(2),
    l_xy_sigma_xy_ratio: float = 2 * sqrt(log(2)),
    l_t_sigma_t_ratio: float = 2 * sqrt(log(2)),
    use_time_frontier_as_centers: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Compute wavelet parameters using linearily-spaced subdivision factors.

    Args:
        order (int): Decomposition order (number of subdivision to perform).
        xx_ref (torch.Tensor): Reference x locations to use to place centers.
        yy_ref (torch.Tensor): Reference y locations to use to place centers.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.
        sigma_t_sigma_xy_ratio (float, optional): Ratio between time gaussian
            enveloppe and space gaussian enveloppe. Default value is set so
            to roughly match 900 km to 20 days and 80 km to 2 days.
            Defaults to 1.1.
        sigma_xy_l_p_ratio (float, optional): Ratio between space gaussian
            enveloppe std and subdivide length. Defaults set so that σ' allows
            exp((-x²-y²)/σ') ~ cos²(πx/(2σ))cos²(πy/(2σ)) close to 0.
            Defaults to 1.75/π*2*√2.
        l_xy_sigma_xy_ratio (float, optional): Ratio between space patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        l_t_sigma_t_ratio (float, optional): Ratio between time patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        use_time_frontier_as_centers (bool, optional): Center gaussian
            enveloppes on time boundaries. If set to True, no guarante that
            time enveloppes will overlap at 0.5. Defaults to False.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    Lx = (xx_ref[-1, 0] - xx_ref[0, 0]).item()
    x0 = xx_ref[0, 0].item()
    Ly = (yy_ref[0, -1] - yy_ref[0, 0]).item()
    y0 = yy_ref[0, 0].item()

    space = {}
    time = {}

    with logger.section(f"Performing {order} linear subdivisions..."):
        for p in range(order):
            # Subdivide Lxy_max
            l_p = Lxy_max / (p + 1)
            # Compute k
            k = 2 * torch.pi / l_p
            # Compute associated std for spatial gaussian enveloppe
            sigma_xy = l_p * sigma_xy_l_p_ratio
            # Compute space patches width
            l_xy = l_xy_sigma_xy_ratio * sigma_xy
            # Compute number of patches requires in both direction
            nx = int(Lx / l_xy) + 1
            ny = int(Ly / l_xy) + 1
            # Compute required offset to center patches on the domain
            offset_x = Lx - (l_xy * (nx - 1))
            offset_y = Ly - (l_xy * (ny - 1))
            # Compute patch centers
            xc = [x0 + offset_x * 0.5 + i * l_xy for i in range(nx)]
            yc = [y0 + offset_y * 0.5 + i * l_xy for i in range(ny)]

            xyc = [(x, y) for x, y in itertools.product(xc, yc)]

            space[p] = {
                "centers": xyc,
                "kx": k,
                "ky": k,
                "sigma_x": sigma_xy,
                "sigma_y": sigma_xy,
                "numel": len(xyc),
            }
            # Compute std for time gaussian enveloppe
            sigma_t = sigma_t_sigma_xy_ratio * sigma_xy
            # Compute associated time patch size
            l_t = l_t_sigma_t_ratio * sigma_t
            # Compute required number of time patches
            nt = int(Lt_max // l_t) + 1

            if use_time_frontier_as_centers:
                # Compute adjusted patch lengths
                l_t = Lt_max / nt
                # Compute patch centers
                tc = [i * l_t for i in range(nt + 1)]
            else:
                # Compute required offset to center patches on the domain
                offset_t = Lt_max - (nt - 1) * l_t
                # Compute patch centers
                tc = [offset_t * 0.5 + i * l_t for i in range(nt)]

            time[p] = {
                "centers": tc,
                "sigma_t": sigma_t,
                "numel": len(tc),
            }

            msg = (
                f"[Subdivision {step(p + 1, order)}] {meters2text(l_xy)} x "
                f"{meters2text(l_xy)} space patches associated to"
                f" {sec2text(l_t)} long time patches."
            )
            logger.detail(msg)
    return space, time


def regular_subdivisions(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Generate space parameters for the Wavelet basis.

    Args:
        order (int): Order of decomposition.
        xx_ref (torch.Tensor): X locations.
        yy_ref (torch.Tensor): Y locations.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    space = {}
    time = {}
    lx = (xx_ref[-1, 0] - xx_ref[0, 0]).cpu().item()
    ly = (yy_ref[0, -1] - yy_ref[0, 0]).cpu().item()
    Lx = lx if Lxy_max is None else Lxy_max
    Ly = ly if Lxy_max is None else Lxy_max
    ratio = 1 / sqrt(log(2))
    for p in range(order):
        Lx_p = Lx / 2**p
        Ly_p = Ly / 2**p
        kx_p = 2 * torch.pi / Lx_p
        ky_p = 2 * torch.pi / Ly_p

        lx_p = lx / 2**p
        xs = [xx_ref[0, 0] + (2 * k + 1) / 2 * lx_p for k in range(2**p)]
        ly_p = ly / 2**p
        ys = [yy_ref[0, 0] + (2 * k + 1) / 2 * ly_p for k in range(2**p)]

        centers = [
            (x.cpu().item(), y.cpu().item())
            for x, y in itertools.product(xs, ys)
        ]

        sigma_x = lx_p / 2 * ratio  # For the gaussian enveloppe
        sigma_y = ly_p / 2 * ratio  # For the gaussian enveloppe

        space[p] = {
            "centers": centers,
            "kx": kx_p,
            "ky": ky_p,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "numel": len(centers),
        }

        lt_p = Lt_max / 2**p
        tc = [(2 * k + 1) / 2 * lt_p for k in range(2**p)]

        sigma_t = lt_p / 2 * ratio
        time[p] = {
            "centers": tc,
            "sigma_t": sigma_t,
            "numel": len(tc),
        }

    return space, time


def _generate_time_params(
    order: int,
    tt: torch.Tensor,
    *,
    Lt_max: float | None = None,
    sigma_ratio: float | None = None,
) -> dict[str, Any]:
    """Generate time parameters for the Wavelet basis.

    Args:
        order (int): Order of decomposition.
        tt (torch.Tensor): Times.
        Lt_max (float | None, optional): Largest dimension along time,
            total width if set to None. Defaults to None.
        sigma_ratio (float | None, optional): Ratio to use to compute sigma,
            if None, set to 1/sqrt(log(2)). Defaults to None.

    Returns:
        dict[str, Any]: Time basis dictionnary.
    """
    basis = {}
    lt = (tt[-1] - tt[0]).cpu().item()
    Lt = lt if Lt_max is None else Lt_max
    tspecs = specs.from_tensor(tt)
    ratio = (
        1 / torch.sqrt(torch.log(torch.tensor(2, **tspecs))).cpu().item()
        if sigma_ratio is None
        else sigma_ratio
    )
    for p in range(order):
        lt_p = Lt / 2**p
        tc = [tt[0] + (2 * k + 1) / 2 * lt_p for k in range(2**p)]

        centers = [t.cpu().item() for t in tc]

        sigma_t = lt_p / 2 * ratio  # For the gaussian enveloppe

        basis[p] = {
            "centers": centers,
            "sigma_t": sigma_t,
            "numel": len(centers),
        }
    return basis


def plot_space_params(space: dict[int, dict[str, Any]]) -> None:
    """Plot space centers.

    Args:
        space (dict[int, dict[str, Any]]): Space parameters dictionnary.
    """
    minmax_x = None
    minmax_y = None
    for v in space.values():
        xyc_p = v["centers"]
        xs = [x for x, _ in xyc_p]
        ys = [y for _, y in xyc_p]

        if minmax_x is None:
            minmax_x = (min(xs), max(xs))
        else:
            minmax_x = (min(*xs, minmax_x[0]), max(*xs, minmax_x[1]))
        if minmax_y is None:
            minmax_y = (min(ys), max(ys))
        else:
            minmax_y = (min(*ys, minmax_y[0]), max(*ys, minmax_y[1]))

    x = np.linspace(minmax_x[0], minmax_x[-1])
    y = np.linspace(minmax_y[0], minmax_y[-1])
    xx, yy = np.meshgrid(x, y, indexing="ij")

    for k, v in space.items():
        xyc_p = v["centers"]
        xs = [x for x, _ in xyc_p]
        ys = [y for _, y in xyc_p]
        sx = v["sigma_x"]
        sy = v["sigma_y"]

        exp_field = np.zeros_like(xx)

        for xc, yc in zip(xs, ys):
            exp_field += np.exp(
                -((xx - xc) ** 2) / sx**2 - (yy - yc) ** 2 / sy**2
            )

        cbar = plt.imshow(exp_field.T, extent=[x[0], x[-1], y[0], y[-1]])
        plt.colorbar(cbar)
        plt.scatter(xs, ys, label=f"{k}", c="red")

        plt.title(f"Order: {k}")
        plt.show()


def plot_time_params(time: dict[int, dict[str, Any]]) -> None:
    """Plot time centers.

    Args:
        time (dict[int, dict[str, Any]]): Time parameters.
    """
    minmax_t = None
    for v in time.values():
        tc_p = v["centers"]
        if minmax_t is None:
            minmax_t = (min(tc_p), max(tc_p))
        else:
            minmax_t = (min(*tc_p, minmax_t[0]), max(*tc_p, minmax_t[1]))
    t = np.linspace(minmax_t[0], minmax_t[1], 200)

    for k, v in time.items():
        tc_p = v["centers"]
        st = v["sigma_t"]
        exp_field = np.zeros_like(t)
        for tc in tc_p:
            exp_field = np.exp(-((t - tc) ** 2) / st**2)
            plt.plot(t, exp_field, zorder=0)
        for tc in tc_p:
            plt.scatter(tc, 0, zorder=1, marker="|", s=200)
        plt.hlines([0.5], [t[0]], [t[-1]], color="grey", linestyles="--")
        plt.legend()
        plt.title(f"Order: {k}")
        plt.show()
