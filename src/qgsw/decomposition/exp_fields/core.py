"""Exponential fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from qgsw import specs
from qgsw.decomposition.supports.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
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
        return self._time["numel"] * self._time["numel"]

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
            self._time["numel"], self._space["numel"], **self._specs
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
        sx: float = self._space["sigma_x"]
        sy: float = self._space["sigma_y"]

        x, y = self._compute_space_params(self._space, xx, yy)

        E = GaussianSupport(x, y, sx, sy)
        e = NormalizedGaussianSupport(E)

        return torch.einsum("tc,cxy->txy", self._coefs, e.field)

    @staticmethod
    def _at_time(
        t: torch.Tensor,
        space_fields: torch.Tensor,
        time_params: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the total field value at a given time.

        Args:
            t (torch.Tensor): Time to compute field at.
            space_fields (dict[int, torch.Tensor]): Space-only fields.
            time_params (dict[int, dict[str, Any]]): Time parameters.

        Returns:
            torch.Tensor: Resulting field.
        """
        tspecs = specs.from_tensor(t)

        tc = time_params["centers"]
        st = time_params["sigma_t"]

        tc = torch.tensor(tc, **tspecs)

        exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)

        return torch.einsum("t,txy->xy", exp, space_fields)

    @staticmethod
    def _dt_at_time(
        t: torch.Tensor,
        space_fields: torch.Tensor,
        time_params: dict[int, dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the total time-derivated field value at a given time.

        Args:
            t (torch.Tensor): Time to compute field at.
            space_fields (dict[int, torch.Tensor]): Space-only fields.
            time_params (dict[int, dict[str, Any]]): Time parameters.

        Returns:
            torch.Tensor: Resulting field.
        """
        tspecs = specs.from_tensor(t)

        tc = time_params["centers"]
        st = time_params["sigma_t"]

        tc = torch.tensor(tc, **tspecs)

        exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
        dt_exp = -2 * (t - tc) / st**2 * exp

        return torch.einsum("t,txy->xy", dt_exp, space_fields)

    def localize(self, xx: torch.Tensor, yy: torch.Tensor) -> EFFunc:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return ExpField._at_time(t, space_fields, self._time)

        return at_time

    def localize_dt(self, xx: torch.Tensor, yy: torch.Tensor) -> EFFunc:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return ExpField._dt_at_time(t, space_fields, self._time)

        return at_time
