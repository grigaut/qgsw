"""Grid conversions."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812


def pool_2d(padded_f: torch.Tensor) -> torch.Tensor:
    """Convolute by summing on 3x3 kernels.

    The Matrix :
    | 0, 0, 0, 0 |
    | 0, a, b, 0 |
    | 0, c, d, 0 |
    | 0, 0, 0, 0 |

    will return :
    |  a ,   a+b  ,   a+b  ,  b  |
    | a+c, a+b+c+d, a+b+c+d, b+c |
    | a+c, a+b+c+d, a+b+c+d, b+c |
    |  c ,   c+d  ,   c+d  ,  d  |

    Args:
        padded_f (torch.Tensor): Tensor to pool.

    Returns:
        torch.Tensor: Padded tensor.
    """
    # average pool padded value
    f_sum_pooled = F.avg_pool2d(
        padded_f,
        (3, 1),
        stride=(1, 1),
        padding=(1, 0),
        divisor_override=1,
    )
    return F.avg_pool2d(
        f_sum_pooled,
        (1, 3),
        stride=(1, 1),
        padding=(0, 1),
        divisor_override=1,
    )


def replicate_pad(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replicate a given pad.

    Args:
        f (torch.Tensor): Tensor to pad.
        mask (torch.Tensor): Mask to use.

    Returns:
        torch.Tensor: Result
    """
    f_ = F.pad(f, (1, 1, 1, 1))
    mask_ = F.pad(mask, (1, 1, 1, 1))
    mask_sum = pool_2d(mask_)
    f_sum = pool_2d(f_)
    f_out = f_sum / torch.maximum(torch.ones_like(mask_sum), mask_sum)
    return mask_ * f_ + (1 - mask_) * f_out


def interpolate(corners: torch.Tensor) -> torch.Tensor:
    """Convert cell corners values to cell centers values.

           Cells corners                            Cells Centers

    x-------x-------x-------x..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   x   |   x   |   x   |
    |       |       |       |                |       |       |       |
    x-------x-------x-------x..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   x   |   x   |   x   |
    |       |       |       |                |       |       |       |
    x-------x-------x-------x..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   x   |   x   |   x   |
    |       |       |       |                |       |       |       |
    x-------x-------x-------x..               ------- ------- ------- ..
    :       :       :       :                :       :       :       :

    Args:
        corners (torch.Tensor): Cells corners values

    Returns:
    torch.Tensor: Cells centers values
    """
    top_left = corners[..., :-1, :-1]
    top_right = corners[..., 1:, :-1]
    bottom_left = corners[..., :-1, 1:]
    bottom_right = corners[..., 1:, 1:]
    return 0.25 * (top_right + top_left + bottom_left + bottom_right)


def interpolate1D(corners: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """Convert cell corners values to cell centers values.

           Cells corners                            Cells Centers

    x-------x-------x-------x..               ---X--- ---X--- ---X--- ..

    Args:
        corners (torch.Tensor): Cells corners values

    Returns:
    torch.Tensor: Cells centers values
    """
    left = corners[..., :, :-1]
    right = corners[..., :, :-1]
    return 0.5 * (left + right)


def omega_to_h(omega_grid: torch.Tensor) -> torch.Tensor:
    """Convert omega grid to h grid.

    Intuitive Grid Representations are :

            omega grid                                 h grid
    y                                        y
    ^                                        ^

    :       :       :       :                :       :       :       :
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   h   |   h   |   h   |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   h   |   h   |   h   |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |   h   |   h   |   h   |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..   > x         ------- ------- ------- ..   > x

    While their actual implementation is:

            omega grid                                  h grid

    ω-------ω-------ω-------ω..   > y          ------- ------- ------- ..   > y
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |   h   |   h   |   h   |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |   h   |   h   |   h   |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |   h   |   h   |   h   |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    :       :       :       :                 :       :       :       :


    v                                         v
    x                                         x

    Args:
        omega_grid (torch.Tensor): omega grid.
            └── (nx+1, ny+1)-shaped

    Returns:
        torch.Tensor: h grid.
            └── (nx, ny)-shaped
    """
    return interpolate(omega_grid)


def omega_to_u(omega_grid: torch.Tensor) -> torch.Tensor:
    """Convert omega grid to u grid.

    Intuitive Grid Representations are :

            omega grid                                 u grid
    y                                        y
    ^                                        ^

    :       :       :       :                :       :       :       :
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                u       u       u       u
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                u       u       u       u
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                u       u       u       u
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..   > x         ------- ------- ------- ..   > x

    While their actual implementation is:

            omega grid                                  u grid

    ω-------ω-------ω-------ω..   > y          ---u--- ---u--- ---u--- ..   > y
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ---u--- ---u--- ---u--- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ---u--- ---u--- ---u--- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ---u--- ---u--- ---u--- ..
    :       :       :       :                 :       :       :       :


    v                                         v
    x                                         x

    Args:
        omega_grid (torch.Tensor): omega grid
            └── (nx+1, ny+1)-shaped

    Returns:
        torch.Tensor: u grid
            └── (nx+1, ny)-shaped
    """
    return 0.5 * (omega_grid[..., :, :-1] + omega_grid[..., :, 1:])


def omega_to_v(omega_grid: torch.Tensor) -> torch.Tensor:
    """Convert omega grid to v grid.

    Intuitive Grid Representations are :

            omega grid                                 v grid
    y                                        y
    ^                                        ^

    :       :       :       :                :       :       :       :
    ω-------ω-------ω-------ω..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    |       |       |       |                |       |       |       |
    ω-------ω-------ω-------ω..   > x         ---v--- ---v--- ---v--- ..   > x

    W ile t eir actual implementation is:

            omega grid                                  v grid

    ω-------ω-------ω-------ω..   > y          ------- ------- ------- ..   > y
    |       |       |       |                 |       |       |       |
    |       |       |       |                 v       v       v       v
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 v       v       v       v
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |       |       |       |                 v       v       v       v
    |       |       |       |                 |       |       |       |
    ω-------ω-------ω-------ω..                ------- ------- ------- ..
    :       :       :       :                 :       :       :       :


    v                                         v
    x                                         x

    Args:
        omega_grid (torch.Tensor): omega grid
            └── (nx+1, ny+1)-shaped

    Returns:
        torch.Tensor: v grid
            └── (nx, ny+1)-shaped
    """
    return 0.5 * (omega_grid[..., :-1, :] + omega_grid[..., 1:, :])


def h_to_u(
    h_grid: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert h grid to u grid.

    Intuitive Grid Representations are :

            h grid                                 u grid
    y                                        y
    ^                                        ^

    :       :       :       :                :       :       :       :
     ------- ------- ------- ..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                u       u       u       u
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                u       u       u       u
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..               ------- ------- ------- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                u       u       u       u
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..   > x         ------- ------- ------- ..   > x

    While their actual implementation is:

            h grid                                  u grid

     ------- ------- ------- ..   > y          ---u--- ---u--- ---u--- ..   > y
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 |       |       |       |
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ---u--- ---u--- ---u--- ..
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 |       |       |       |
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ---u--- ---u--- ---u--- ..
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 |       |       |       |
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ---u--- ---u--- ---u--- ..
    :       :       :       :                 :       :       :       :


    v                                         v
    x                                         x

    Args:
        h_grid (torch.Tensor): h grid
            └── (nx, ny)-shaped
        mask (torch.Tensor|None): Mask tensor, if None, the mask
        will be considerated as equal to 1 evry points of the domain.,
        Defaults to None.

    Returns:
        torch.Tensor: u grid
            └── (nx+1, ny)-shaped
    """
    if mask is None:
        mask = torch.ones_like(h_grid)
    extended_h = replicate_pad(h_grid, mask)
    return 0.5 * (extended_h[..., 1:, 1:-1] + extended_h[..., :-1, 1:-1])


def h_to_v(
    h_grid: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert h grid to v grid.

    Intuitive Grid Representations are :

            h grid                                 v grid
    y                                        y
    ^                                        ^

    :       :       :       :                :       :       :       :
     ------- ------- ------- ..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                |       |       |       |
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                |       |       |       |
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..               ---v--- ---v--- ---v--- ..
    |       |       |       |                |       |       |       |
    |   h   |   h   |   h   |                |       |       |       |
    |       |       |       |                |       |       |       |
     ------- ------- ------- ..   > x         ---v--- ---v--- ---v--- ..   > x

    While their actual implementation is:

            h grid                                  v grid

     ------- ------- ------- ..   > y          ------- ------- ------- ..   > y
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 v       v       v       v
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 v       v       v       v
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ------- ------- ------- ..
    |       |       |       |                 |       |       |       |
    |   h   |   h   |   h   |                 v       v       v       v
    |       |       |       |                 |       |       |       |
     ------- ------- ------- ..                ------- ------- ------- ..
    :       :       :       :                 :       :       :       :


    v                                         v
    x                                         x

    Args:
        h_grid (torch.Tensor): h grid
            └── (nx, ny)-shaped
        mask (torch.Tensor|None): Mask tensor, if None, the mask
        will be considerated as equal to 1 evry points of the domain.,
        Defaults to None.

    Returns:
        torch.Tensor: v grid
            └── (nx, ny+1)-shaped
    """
    if mask is None:
        mask = torch.ones_like(h_grid)
    extended_h = replicate_pad(h_grid, mask)
    return 0.5 * (extended_h[..., 1:-1, 1:] + extended_h[..., 1:-1, :-1])
