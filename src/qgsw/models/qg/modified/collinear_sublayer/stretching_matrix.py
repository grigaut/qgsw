"""Stretching matrix for collinear Stream Function."""

import torch

from qgsw.models.qg.stretching_matrix import compute_A


def compute_A_collinear_sf(  # noqa: N802
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
    alpha: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute new Stretching operator.

    Ã = (1/ρ_1)[[(1/H_1)*(1/g_1 + (1 - α)/g_2)]]

    Args:
        H (torch.Tensor): Layers reference height.
        g_prime (torch.Tensor): Reduced gravity values.
        alpha (float): Collinearity coefficient.
        dtype (torch.dtype): Data type.
        device: (torch.device): Data device.

    Returns:
        torch.Tensor: Stretching Operator
    """  # noqa: RUF002
    A = compute_A(H=H, g_prime=g_prime, dtype=dtype, device=device)  # noqa: N806
    # Create layers coefficients vector [1, α]  # noqa: RUF003
    layers_coefs = torch.tensor(
        [1, alpha],
        dtype=dtype,
        device=device,
    )
    # Select top row from matrix product
    return (A @ layers_coefs)[0, ...].unsqueeze(0).unsqueeze(0)
