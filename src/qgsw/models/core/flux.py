# ruff: noqa
"""
Velocity-sign biased flux computations.
Louis Thiry, 2023
"""

import torch.nn.functional as F
import torch
from typing import Callable


def stencil_2pts(
    q: torch.Tensor, dim: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 2 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-1, :],
        q[1:, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-1],
        q[:, 1:],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.

    Returns:
        tuple[ torch.Tensor, torch.Tensor, ]: Narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 1),
        q.narrow(dim, 1, n - 1),
    )


def stencil_4pts(
    q: torch.Tensor, dim: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 4 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-3, :],
        q[1:-2, :],
        q[2:-1, :],
        q[3:-2, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-3],
        q[:, 1:-2],
        q[:, 2:-1],
        q[:, 3:-2],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.

    Returns:
        tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: Narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 3),
        q.narrow(dim, 1, n - 3),
        q.narrow(dim, 2, n - 3),
        q.narrow(dim, 3, n - 3),
    )


def stencil_6pts(
    q: torch.Tensor, dim: int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 6 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-5, :],
        q[1:-4, :],
        q[2:-3, :],
        q[3:-2, :],
        q[4:-1, :],
        q[5:, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-5],
        q[:, 1:-4],
        q[:, 2:-3],
        q[:, 3:-2],
        q[:, 4:-1],
        q[:, 5:],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.

    Returns:
        tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: Narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 5),
        q.narrow(dim, 1, n - 5),
        q.narrow(dim, 2, n - 5),
        q.narrow(dim, 3, n - 5),
        q.narrow(dim, 4, n - 5),
        q.narrow(dim, 5, n - 5),
    )


def flux(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
    n_points: int,
    rec_func_2: Callable,
    rec_func_4: Callable,
    rec_func_6: Callable,
    mask_2: torch.Tensor,
    mask_4: torch.Tensor,
    mask_6: torch.Tensor,
):
    # positive and negative velocities
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # 2-points reconstruction
    q_stencil2 = stencil_2pts(q, dim)
    # Positive velocities: use direct stencil
    qi2_pos = rec_func_2(*q_stencil2)
    # Negative velocities: use reversed stencil
    qi2_neg = rec_func_2(*q_stencil2[::-1])

    if n_points == 2:
        return u_pos * qi2_pos + u_neg * qi2_neg

    # 4-points reconstruction
    pad = (1, 1, 0, 0) if dim == -1 else (0, 0, 1, 1)
    q_stencil4 = stencil_4pts(q, dim)
    qi4_pos = F.pad(rec_func_4(*q_stencil4), pad)
    qi4_neg = F.pad(rec_func_4(*q_stencil4[::-1]), pad)

    if n_points == 4:
        return u_pos * (mask_2 * qi2_pos + mask_4 * qi4_pos) + u_neg * (
            mask_2 * qi2_neg + mask_4 * qi4_neg
        )

    # 6-points reconstruction
    pad = (2, 2, 0, 0) if dim == -1 else (0, 0, 2, 2)
    q_stencil6 = stencil_6pts(q, dim)
    qi6_pos = F.pad(rec_func_6(*q_stencil6), pad)
    qi6_neg = F.pad(rec_func_6(*q_stencil6[::-1]), pad)

    if n_points == 6:
        return u_pos * (
            mask_2 * qi2_pos + mask_4 * qi4_pos + mask_6 * qi6_pos
        ) + u_neg * (mask_2 * qi2_neg + mask_4 * qi4_neg + mask_6 * qi6_neg)

    # raise NotImplementedError(f'flux computations implemented for '
    # f'2, 4, 6 points stencils, got {n_points}')
