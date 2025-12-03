"""Exponential fields."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from math import log, sqrt
from typing import Any

import torch

from qgsw import specs
from qgsw.decomposition.wavelets.supports import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qgsw.specs import defaults

EFFunc = Callable[[torch.Tensor], torch.Tensor]


def subdivisions(
    xx_ref: torch.Tensor, yy_ref: torch.Tensor
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute space subdivision in 4.

    Args:
        xx_ref (torch.Tensor): Refrecence X locations.
        yy_ref (torch.Tensor): Reference Y locations.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Space params, time params.
    """
    o = 4

    Lx = xx_ref[-1, 0] - xx_ref[0, 0]
    Ly = yy_ref[0, -1] - yy_ref[0, 0]

    lx = Lx / o
    ly = Ly / o
    lt = 20 * 3600 * 24 / o

    xc = [(2 * k + 1) * lx / 2 for k in range(o)]
    yc = [(2 * k + 1) * ly / 2 for k in range(o)]

    centers = [
        (x.cpu().item(), y.cpu().item()) for x, y in itertools.product(xc, yc)
    ]

    tc = [(2 * k + 1) * lt / 2 for k in range(o)]

    sx = lx / 2 / sqrt(log(2))
    sy = ly / 2 / sqrt(log(2))
    st = lt / 2 / sqrt(log(2))

    return {
        "centers": centers,
        "sigma_x": sx,
        "sigma_y": sy,
        "numel": len(centers),
    }, {
        "centers": tc,
        "sigma_t": st,
        "numel": len(tc),
    }


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
