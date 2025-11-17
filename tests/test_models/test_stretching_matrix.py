"""Stretching matrix tests."""

import torch

from qgsw import specs
from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_deformation_radii,
)


def test_deformation_radius() -> None:
    """Test deformation radius computation."""
    H1, H2 = 100, 200
    Htot = H1 + H2
    Heq = H1 * H2 / Htot
    g1, g2 = 10, 0.1
    f0 = 1e-4

    H = torch.tensor([H1, H2], dtype=torch.float64)
    g = torch.tensor([g1, g2], dtype=torch.float64)

    A = compute_A(H, g, **specs.from_tensor(H))
    Ld = compute_deformation_radii(A, f0)

    Ld_ref = (
        torch.tensor([Heq * g2, Htot * g1], dtype=torch.float64)
    ).sqrt() / f0
    assert ((Ld - Ld_ref) <= (1e-5 + 1e-2 * torch.abs(Ld_ref))).all()
    torch.testing.assert_close(Ld, Ld_ref, rtol=1e-2, atol=1e-5)
