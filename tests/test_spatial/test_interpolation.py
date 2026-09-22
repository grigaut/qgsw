"""Test interpolations."""

import torch

from qgsw.spatial.core.grid_conversion import interpolate1D


def test_interpolation1D() -> None:  # noqa: N802
    """Test 1D interpolation."""
    t = torch.rand((50, 50))

    t0 = interpolate1D(t, dim=0)

    torch.testing.assert_close(t0, (t[1:, :] + t[:-1, :]) / 2)

    t_2 = interpolate1D(t, dim=-2)

    torch.testing.assert_close(t_2, (t[1:, :] + t[:-1, :]) / 2)

    t1 = interpolate1D(t, dim=1)

    torch.testing.assert_close(t1, (t[:, 1:] + t[:, :-1]) / 2)

    t_1 = interpolate1D(t, dim=-1)

    torch.testing.assert_close(t_1, (t[:, 1:] + t[:, :-1]) / 2)
