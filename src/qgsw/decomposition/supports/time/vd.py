"""Compact-support (cos²) time support for wavelet bases."""

import math

import torch

from qgsw.decomposition.supports.space.vd import (
    SupportLayout,
)
from qgsw.decomposition.supports.time.base import TimeSupportFunction


def window(t: torch.Tensor) -> torch.Tensor:
    """Window function."""
    return torch.cos(0.5 * math.pi * t).square()


def window_dt(t: torch.Tensor) -> torch.Tensor:
    """Time derivative of window function."""
    return -0.5 * math.pi * torch.sin(math.pi * t)


class VDTimeSupport(TimeSupportFunction):
    """cos² time support with per-wavelet decorrelation time.

    time_params[lvl] = {
        "enst": (NP, Nt) array, NaN where a wavelet has no time center,
        "tdec": (NP,), "norm_fact": (NP,),
        "layout": SupportLayout of the matching space function,
    }
    space_fields[lvl] : (nsupp, Nt), i.e. space.project(coefs, operator).
    """

    def __init__(
        self,
        time_params: dict[int, dict],
        space_fields: dict[int, torch.Tensor],
        layouts: dict[int, SupportLayout],
        space_shape: tuple[int, ...] = (-1,),
    ) -> None:
        """Instantiate the VD time support.

        Args:
            time_params (dict[int, dict]): Time parameters.
            space_fields (dict[int, torch.Tensor]): Space fields.
            layouts (dict[int, SupportLayout]): Layouts.
            space_shape (tuple[int, ...], optional): Space shape
                (for reshaping). Defaults to (-1,).
        """
        self._space = space_fields
        self.params = time_params
        self.shape = space_shape
        self._cache = {}
        for lvl, p in time_params.items():
            ref = space_fields[lvl]
            spec = {"dtype": ref.dtype, "device": ref.device}
            centers = torch.as_tensor(p["centers"], **spec)
            nan = torch.isnan(centers)
            layout = layouts[lvl]
            self._cache[lvl] = {
                "centers": torch.where(
                    nan, torch.zeros_like(centers), centers
                ),
                "nan": nan,
                "tdec": torch.as_tensor(p["tdec"], **spec)[:, None],
                "norm": torch.as_tensor(p["norm"], **spec)[:, None],
                "ip": layout.ip,
                "ix": layout.ix,
                "nphys": layout.nphys,
            }

    def retrieve_support(self, lvl: int, t: torch.Tensor) -> torch.Tensor:
        """Time factors, shape (*t.shape, NP, Nt)."""
        d = self._cache[lvl]
        x = (t[..., None, None] - d["centers"]) / d["tdec"]
        valid = (x.abs() <= 1.0) & ~d["nan"]
        f = window(x)
        return torch.where(valid, f / d["norm"], torch.zeros_like(f))

    def retrieve_dt_support(self, lvl: int, t: torch.Tensor) -> torch.Tensor:
        """Time factors, shape (*t.shape, NP, Nt)."""
        d = self._cache[lvl]
        x = (t[..., None, None] - d["centers"]) / d["tdec"]
        valid = (x.abs() <= 1.0) & ~d["nan"]
        f = window_dt(x) / d["tdec"]
        return torch.where(valid, f / d["norm"], torch.zeros_like(f))

    def retrieve_supports(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Retrieve supports."""
        return {lvl: self.retrieve_support(lvl, t) for lvl in self.params}

    def retrieve_dt_supports(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Retrieve dt supports."""
        return {lvl: self.retrieve_dt_support(lvl, t) for lvl in self.params}

    def _contract(self, lvl: int, w: torch.Tensor) -> torch.Tensor:
        """Sum over (e, p) then scatter on grid points: (*t.shape, nphys)."""
        d = self._cache[lvl]
        Y = self._space[lvl]  # (nsupp, Nt)
        contrib = torch.einsum("...se,se->...s", w[..., d["ip"], :], Y)
        out = contrib.new_zeros(*contrib.shape[:-1], d["nphys"])
        return out.index_add(contrib.dim() - 1, d["ix"], contrib)

    def decompose(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Decompose time fields over scales.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            dict[int, torch.Tensor]: Mapping.
        """
        return {
            lvl: self._contract(lvl, self.retrieve_support(lvl, t)).reshape(
                (-1, *self.shape)
            )
            for lvl in self.params
        }

    def decompose_dt(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Decompose dt fields over scales.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            dict[int, torch.Tensor]: Mapping.
        """
        return {
            lvl: self._contract(lvl, self.retrieve_dt_support(lvl, t)).reshape(
                (-1, *self.shape)
            )
            for lvl in self.params
        }

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate function.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Field at t.
        """
        return sum(self.decompose(t).values())

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate function's time derivative..

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Field at t.
        """
        return sum(self.decompose_dt(t).values())
