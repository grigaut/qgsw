"""Test QG models."""

import pytest
import torch

from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_layers_to_mode_decomposition,
)
from qgsw.specs import DEVICE


@pytest.mark.parametrize(
    ("H", "g_prime", "reference"),
    [
        (
            torch.tensor([100]),
            torch.tensor([10]),
            torch.tensor(
                [[1 / (100 * 10)]],
                dtype=torch.float64,
                device=DEVICE.get(),
            ),
        ),
        (
            torch.tensor([200, 800]),
            torch.tensor([10, 0.05]),
            torch.tensor(
                [
                    [1 / (200 * 10) + 1 / (200 * 0.05), -1 / (200 * 0.05)],
                    [-1 / (800 * 0.05), 1 / (800 * 0.05)],
                ],
                dtype=torch.float64,
                device=DEVICE.get(),
            ),
        ),
    ],
)
def test_stretching_matrix(
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    """Test streching matrix computation."""
    A = compute_A(  # noqa: N806
        H,
        g_prime,
        torch.float64,
        DEVICE.get(),
    )
    assert torch.isclose(reference, A).all()


@pytest.mark.parametrize(
    ("H", "g_prime"),
    [
        (torch.tensor([100]), torch.tensor([10])),
        (torch.tensor([200, 800]), torch.tensor([10, 0.05])),
    ],
)
def test_layer_to_mode_decomposition(
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
) -> None:
    """Test layer to mode decomposition shapes."""
    A = compute_A(H, g_prime, torch.float64, DEVICE.get())  # noqa: N806
    Cm2l, lambd, Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
    assert Cm2l.shape == (H.shape[0], H.shape[0])
    assert lambd.shape == H.shape
    assert Cl2m.shape == (H.shape[0], H.shape[0])
