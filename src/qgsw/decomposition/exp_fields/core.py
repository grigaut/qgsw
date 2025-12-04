"""Exponential fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from qgsw.decomposition.supports.space.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qgsw.decomposition.supports.time.gaussian import GaussianTimeSupport
from qgsw.specs import defaults

EFFunc = Callable[[torch.Tensor], torch.Tensor]


class ExpField:
    """Field of exponentials."""

    def __init__(
        self,
        space_params: dict[str, Any],
        time_params: dict[str, Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the Wavelet basis.

        Args:
            space_params (dict[str, Any]): Space parameters.
            time_params (dict[str, Any]): Time parameters.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Ddevice.
                Defaults to None.
        """
        self._order = len(space_params.keys())
        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params

    def numel(self) -> int:
        """Total number of elements."""
        return self._time["numel"] * self._space["numel"]

    def _compute_space_params(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = params["centers"]
        xc = torch.tensor([c[0] for c in centers], **self._specs)
        yc = torch.tensor([c[1] for c in centers], **self._specs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]
        return x, y

    def generate_random_coefs(self) -> torch.Tensor:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            torch.Tensor: Coefficients.
        """
        return torch.randn(
            self._time[0]["numel"], self._space[0]["numel"], **self._specs
        )

    def set_coefs(self, coefs: torch.Tensor) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (torch.Tensor): Coefficients.
        """
        self._coefs = coefs

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build space-related fields.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        sx: float = self._space[0]["sigma_x"]
        sy: float = self._space[0]["sigma_y"]

        x, y = self._compute_space_params(self._space[0], xx, yy)

        E = GaussianSupport(x, y, sx, sy)
        e = NormalizedGaussianSupport(E)

        return {0: torch.einsum("tc,cxy->txy", self._coefs, e.field)}

    def localize(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> GaussianTimeSupport:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            GaussianTimeSuport: Function computing the wavelet field at
                a given time.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        return GaussianTimeSupport(self._time, space_fields)
