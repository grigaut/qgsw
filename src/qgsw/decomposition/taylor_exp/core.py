"""Taylor series time support with exponential space supports."""

from typing import Any

import torch
from torch import Tensor

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.supports.space.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.supports.time.taylor import TaylorSeriesTimeSupport


class TaylorExpBasis(
    SpaceTimeDecomposition[NormalizedGaussianSupport, TaylorSeriesTimeSupport]
):
    """Taylor Series Time suuport and Gaussian space support."""

    type = "taylor-exps"

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

    def generate_random_coefs(self) -> DecompositionCoefs:  # noqa: D102
        coefs = {}
        for k in self._space:
            coefs[k] = torch.randn(
                (
                    self._time[k]["numel"],
                    self._space[k]["numel"],
                ),
                **self._specs,
            )

        return DecompositionCoefs.from_dict(coefs)

    generate_random_coefs.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, TaylorSeriesTimeSupport
    ].generate_random_coefs.__doc__

    def generate_time_support(  # noqa: D102
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, Tensor],
    ) -> TimeSupportFunction:
        return TaylorSeriesTimeSupport(time_params, space_fields)

    generate_time_support.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, TaylorSeriesTimeSupport
    ].generate_time_support.__doc__

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

    def _build_space_dx(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx)
        return fields

    def _build_space_dy(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy)
        return fields

    def _build_space_dx2(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx2)
        return fields

    def _build_space_dy2(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy2)
        return fields

    def _build_space_dx3(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx3)
        return fields

    def _build_space_dy3(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy3)
        return fields

    def _build_space_dydx2(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dydx2)
        return fields

    def _build_space_dxdy2(
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
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dxdy2)
        return fields
