"""PV computing functions."""

from typing import Callable

import torch

from qgsw.pv import compute_q1_interior, compute_q1_interior_bcs
from qgsw.scripts.boundaries import extract_q_bc
from qgsw.solver.boundary_conditions.base import Boundaries


def build_pv_funcs(
    dx: float, dy: float, f0: float, beta_effect: torch.Tensor, b: int, nx: int
) -> Callable[
    [torch.Tensor, torch.Tensor],
    tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[Boundaries], Boundaries],
    ],
]:
    """Build function to compute PV values and BCs.

    Args:
        dx (float): dx.
        dy (float): dy.
        f0 (float): f0.
        beta_effect (torch.Tensor): beta effect tensor
        b (int): Boundary width.
        nx (int): Space width.

    Returns:
        tuple[ Callable[ [torch.Tensor, torch.Tensor], Callable[[torch.Tensor], torch.Tensor] ], Callable[[torch.Tensor, torch.Tensor], Callable[[Boundaries], Boundaries]], ]: _description_
    """  # noqa: E501

    def compute_pv(
        A11: torch.Tensor, A12: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda psi1: compute_q1_interior(
            psi1,
            torch.zeros_like(psi1),
            A11,
            A12,
            dx,
            dy,
            f0,
            beta_effect[:, 1:-1],
        )

    def compute_pv_bc(
        A11: torch.Tensor, A12: torch.Tensor
    ) -> Callable[
        [torch.Tensor, torch.Tensor], Callable[[Boundaries], Boundaries]
    ]:
        return lambda psi1: compute_q1_interior_bcs(
            psi1,
            Boundaries.zeros_like(psi1),
            A11,
            A12,
            dx,
            dy,
            f0,
            extract_q_bc(beta_effect[:, 1:-1].tile((nx - 2, 1)), b),
        )

    return lambda A11, A12: (compute_pv(A11, A12), compute_pv_bc(A11, A12))
