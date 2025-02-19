"""Special variables."""

from collections.abc import Callable

import torch

from qgsw.fields.variables.prognostic_tuples import UVH
from qgsw.filters.base import _Filter
from qgsw.utils.shape_checks import with_shapes


@with_shapes(g_prime=(2,))
def compute_g_tilde(g_prime: torch.Tensor) -> torch.Tensor:
    """Compute g_tilde = g_1 g_2 / (g_1 + g_2).

    Args:
        g_prime (torch.Tensor): Reduced gravity tensor.
            └── (2,) shaped

    Returns:
        torch.Tensor: g_tilde = g_1 g_2 / (g_1 + g_2)
            └── (1,) shaped
    """
    if g_prime.shape != (2,):
        msg = f"g' should be (2,)-shaped, not {g_prime.shape}."
        raise ValueError(msg)
    g1, g2 = g_prime
    return (g1 * g2 / (g1 + g2)).unsqueeze(0)


@with_shapes(
    alpha=(1,),
    H1=(1,),
    g2=(1,),
)
def compute_source_term_factor(
    alpha: torch.Tensor,
    H1: torch.Tensor,  # noqa: N803
    g2: torch.Tensor,
    f0: float,
) -> torch.Tensor:
    """Compute source term multiplicative factor.

    Args:
        alpha (torch.Tensor): Collinearity coefficient.
            └── (1, )-shaped.
        H1 (torch.Tensor): Top layer reference depth.
            └── (1, )-shaped.
        g2 (torch.Tensor): Reduced gravity in the second layer.
            └── (1, )-shaped.
        f0 (float): f0.

    Returns:
        torch.Tensor: f_0²α/H1/g2
    """
    return f0**2 * alpha / H1 / g2


@with_shapes(
    alpha=(1,),
    H1=(1,),
    g2=(1,),
    g_tilde=(1,),
)
def compute_source_term(
    uvh: UVH,
    filt: _Filter,
    alpha: torch.Tensor,
    H1: torch.Tensor,  # noqa: N803
    g2: torch.Tensor,
    g_tilde: torch.Tensor,
    f0: float,
    points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute source term.

    Args:
        uvh (UVH): Prognostic u,v and h.
            ├── u: (n_ens, 1, nx+1, ny)-shaped
            ├── v: (n_ens, 1, nx, ny+1)-shaped
            └── h: (n_ens, 1, nx, ny)-shaped
        filt (_Filter): Filter.
        alpha (torch.Tensor): Collinearity coefficient.
        H1 (torch.Tensor): Top layer depth.
            └── (1, )-shaped.
        g2 (torch.Tensor): Reduced gravity in the bottom layer.
            └── (1, )-shaped.
        g_tilde (torch.Tensor): Equivalent reduced gravity.
            └── (1, )-shaped.
        f0 (float): f0.
        points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
        to surface interpolation function.

    Returns:
        torch.Tensor: Source term: f_0αg̃/H1/g2/ds (F^s)⁻¹{K F{h}}.
    """
    h_top_i = points_to_surfaces(uvh.h[0, 0])
    h_filt = filt(h_top_i).unsqueeze(0).unsqueeze(0)
    psi = g_tilde * h_filt / f0
    factor = compute_source_term_factor(alpha, H1, g2, f0)
    return factor * psi
