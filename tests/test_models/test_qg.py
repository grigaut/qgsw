"""Test QG models."""

import pytest
import torch

from qgsw.models.qg.stretching_matrix import compute_A


@pytest.mark.parametrize(
    ("H", "g_prime", "reference"),
    [
        (
            torch.tensor([100]),
            torch.tensor([10]),
            torch.tensor(
                [[1 / (100 * 10)]],
                dtype=torch.float64,
                device="cpu",
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
                device="cpu",
            ),
        ),
    ],
)
def test_stretching_matrix(
    H: torch.dtype,  # noqa: N803
    g_prime: torch.dtype,
    reference: torch.Tensor,
) -> None:
    """Test streching matrix computation."""
    A = compute_A(  # noqa: N806
        H,
        g_prime,
        torch.float64,
        "cpu",
    )
    assert torch.isclose(reference, A).all()
