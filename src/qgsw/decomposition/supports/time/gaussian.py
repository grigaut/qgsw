"""Gaussian time supports."""

# ruff: noqa: D107, D102
from typing import Any

import torch
from torch._tensor import Tensor

from qgsw import specs
from qgsw.decomposition.supports.time.base import TimeSupportFunction


class GaussianTimeSupport(TimeSupportFunction):
    """Gaussian time support."""

    def __init__(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, Tensor],
    ) -> None:
        self._space = space_fields
        self.params = time_params

    __init__.__doc__ = TimeSupportFunction.__init__.__doc__

    def decompose(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_ = exp / exp.sum(dim=0)

            fields[lvl] = torch.einsum("t,txy->xy", exp_, self._space[lvl])
        return fields

    decompose.__doc__ = TimeSupportFunction.decompose.__doc__

    def __call__(self, t: Tensor) -> Tensor:
        fields = self.decompose(t)
        return sum(fields.values()) / len(self.params)

    __call__.__doc__ = TimeSupportFunction.__call__.__doc__

    def decompose_dt(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]
            space = self._space[lvl]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_s = exp.sum(dim=0)
            dt_exp = -2 * (t - tc) / st**2 * exp

            dt_e = (dt_exp * exp_s - exp * dt_exp.sum(dim=0)) / exp_s**2

            fields[lvl] = torch.einsum("t,txy->xy", dt_e, space)
        return fields

    decompose_dt.__doc__ = TimeSupportFunction.decompose_dt.__doc__

    def dt(self, t: Tensor) -> Tensor:
        fields = self.decompose_dt(t)
        return sum(fields.values()) / len(self.params)

    dt.__doc__ = TimeSupportFunction.dt.__doc__
