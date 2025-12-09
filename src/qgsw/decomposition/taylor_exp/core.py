"""Taylor series time support with exponential space supports."""

from typing import Any

import torch
from torch import Tensor

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qgsw.decomposition.supports.time.taylor import TaylorSeriesTimeSupport


class TaylorExpBasis(
    SpaceTimeDecomposition[NormalizedGaussianSupport, TaylorSeriesTimeSupport]
):
    """Taylor Series Time suuport and Gaussian space support."""

    def _compute_space_params(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        centers = params["centers"]
        xc = torch.tensor([c[0] for c in centers], **self._specs)
        yc = torch.tensor([c[1] for c in centers], **self._specs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]
        return (x, y)

    def numel(self) -> int:  # noqa: D102
        return sum(s["numel"] for s in self._space.values())

    numel.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, TaylorSeriesTimeSupport
    ].numel.__doc__

    def generate_random_coefs(self) -> dict[int, Tensor]:  # noqa: D102
        coefs = {}
        for k in self._space:
            coefs[k] = torch.randn(
                (
                    self._time[k]["numel"],
                    self._space[k]["numel"],
                ),
                **self._specs,
            )

        return coefs

    numel.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, TaylorSeriesTimeSupport
    ].generate_random_coefs.__doc__

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.field)
        return fields

    def localize(self, xx: Tensor, yy: Tensor) -> TaylorSeriesTimeSupport:  # noqa: D102
        space_fields = self._build_space(xx=xx, yy=yy)

        return TaylorSeriesTimeSupport(self._time, space_fields)

    localize.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, TaylorSeriesTimeSupport
    ].localize.__doc__
