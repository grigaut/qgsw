"""Tests for wavelets."""

import pytest
import torch

from qgsw.decomposition.wavelets import WaveletBasis
from qgsw.specs import defaults

specs = defaults.get()

testdata = [
    pytest.param(1, id="order-1"),
    pytest.param(2, id="order-2"),
    pytest.param(3, id="order-3"),
    pytest.param(4, id="order-4"),
]


@pytest.mark.parametrize(("order"), testdata)
def test_dt(order: int) -> None:
    """Test time derivation."""
    x = torch.arange(0, 100, **specs)
    y = torch.arange(0, 100, **specs)
    x, y = torch.meshgrid(x, y, indexing="ij")
    xx: torch.Tensor = x - x[:1, :]
    yy = y - y[:, :1]
    tt = torch.arange(0, 250, dtype=torch.float64)

    basis = WaveletBasis.from_dyadic_decomposition(
        order, xx, yy, (x[-1, 0] - x[0, 0]).item(), tt[-1].item()
    )
    coefs = basis.generate_random_coefs()
    basis.set_coefs(coefs)

    wv = basis.localize(xx, yy)
    wv_dt = basis.localize_dt(xx, yy)

    dt_discrete = wv(torch.tensor([1], **specs)) - wv(
        torch.tensor([0], **specs)
    )
    dt_analytic = wv_dt(torch.tensor([0.5], **specs))

    torch.testing.assert_close(dt_discrete, dt_analytic, atol=1e-5, rtol=1e-5)
