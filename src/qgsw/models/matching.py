"""Match models variables."""

import torch

from qgsw.fields.variables.uvh import UVH


def n_layers_to_collinear_1_layer(
    uvh_nl: UVH,
    g_prime_nl: torch.Tensor,
    alpha: torch.Tensor,
    g_prime: torch.Tensor,
) -> UVH:
    """Match n-layers uvh to 1 collinear layer.

    Args:
        uvh_nl (UVH): n-layers uvh.
        g_prime_nl (torch.Tensor): g' for the n-layers model.
        alpha (torch.Tensor): Collinearity coefficient.
        g_prime (torch.Tensor): g' for the 1-layer collinear model.

    Raises:
        ValueError: If g' shape is not (2,)

    Returns:
        UVH: 1-layer uvh
    """
    if g_prime.shape != (2,):
        msg = "g' is expected to be (2,)-shaped"
        raise ValueError(msg)
    g1 = g_prime[0]
    g2 = g_prime[1]
    g1_nl = g_prime_nl[0]

    h_tot = torch.sum(uvh_nl.h, dim=-3, keepdim=True)
    h = h_tot * g1_nl * (g2 + (1 - alpha) * g1) / (g1 * g2)

    return UVH(uvh_nl.u[:, :1, ...], uvh_nl.v[:, :1, ...], h)


def n_layers_to_1_layer(
    uvh_nl: UVH,
    g_prime_nl: torch.Tensor,
    g_prime: torch.Tensor,
) -> UVH:
    """Match n-layers uvh to 1 collinear layer.

    Args:
        uvh_nl (UVH): n-layers uvh.
        g_prime_nl (torch.Tensor): g' for the n-layers model.
        g_prime (torch.Tensor): g' for the 1-layer collinear model.

    Raises:
        ValueError: If g' shape is not (1,)

    Returns:
        UVH: 1-layer uvh
    """
    if g_prime.shape != (1,):
        msg = "g' is expected to be (1,)-shaped"
        raise ValueError(msg)
    g1 = g_prime[0]
    g1_nl = g_prime_nl[0]

    h_tot = torch.sum(uvh_nl.h, dim=-3, keepdim=True)
    h = h_tot * g1_nl / g1

    return UVH(uvh_nl.u[:, :1, ...], uvh_nl.v[:, :1, ...], h)
