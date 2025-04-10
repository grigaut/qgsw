"""Special variables."""

import torch

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
