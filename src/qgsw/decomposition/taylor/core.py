"""Core class for Taylor Series time support."""

from typing import Any

import torch

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.supports.space.full_field import FullFieldSpaceSupport
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.supports.time.taylor import TaylorSeriesTimeSupport


class TaylorFullFieldBasis(
    SpaceTimeDecomposition[FullFieldSpaceSupport, TaylorSeriesTimeSupport]
):
    """Taylor Series time support with full field."""

    type = "taylor-fullfield"

    def numel(self) -> int:  # noqa: D102
        return sum(s["numel"] for s in self._space.values())

    numel.__doc__ = SpaceTimeDecomposition[
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
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
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
    ].generate_random_coefs.__doc__

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        nx, ny = xx.shape[-2:]

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            Nx, Ny = params["xs"].shape[0], params["ys"].shape[0]
            mask = self._compute_mask(params, xx, yy)[None, ...]

            full_field = FullFieldSpaceSupport(nx, ny, **self._specs)
            coefs = c.reshape((-1, Nx, Ny))[mask].reshape((-1, nx, ny))
            fields[lvl] = torch.einsum(
                "txy,cxy->txy",
                coefs,
                full_field.field,
            )
        return fields

    def _compute_mask(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> torch.Tensor:
        xs = params["xs"]
        ys = params["ys"]

        x_idx = torch.searchsorted(xs, xx)
        y_idx = torch.searchsorted(ys, yy)
        if not torch.all(xs[x_idx] == xx):
            msg = "x contains values that do not match xs."
            raise ValueError(msg)
        if not torch.all(ys[y_idx] == yy):
            msg = "y contains values that do not match ys."
            raise ValueError(msg)

        mask = torch.zeros(
            (xs.shape[0], ys.shape[0]),
            dtype=torch.bool,
            device=self._specs["device"],
        )
        mask[x_idx, y_idx] = True

        return mask

    def generate_time_support(  # noqa: D102
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, torch.Tensor],
    ) -> TimeSupportFunction:
        return TaylorSeriesTimeSupport(time_params, space_fields)

    generate_time_support.__doc__ = SpaceTimeDecomposition[
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
    ].generate_time_support.__doc__
