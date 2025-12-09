"""Core class for Taylor Series time support."""

import torch

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.full_field import FullFieldSpaceSupport
from qgsw.decomposition.supports.time.taylor import TaylorSeriesTimeSupport


class TaylorFullFieldBasis(
    SpaceTimeDecomposition[FullFieldSpaceSupport, TaylorSeriesTimeSupport]
):
    """Taylor Series time support with full field."""

    def numel(self) -> int:  # noqa: D102
        return sum(s["numel"] for s in self._space.values())

    numel.__doc__ = SpaceTimeDecomposition[
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
    ].numel.__doc__

    def generate_random_coefs(self) -> dict[int, torch.Tensor]:  # noqa: D102
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
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
    ].generate_random_coefs.__doc__

    def _build_space(self) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            nx: float = params["nx"]
            ny: float = params["ny"]

            full_field = FullFieldSpaceSupport(nx, ny, **self._specs)
            fields[lvl] = torch.einsum(
                "txy,cxy->txy",
                c.reshape((-1, nx, ny)),
                full_field.field,
            )
        return fields

    def _check_locations(self, xx: torch.Tensor, yy: torch.Tensor) -> None:
        nx = self._space[0]["nx"]
        ny = self._space[0]["ny"]
        if (s := xx.shape) != (nx, ny):
            msg = f"X-locations must be {nx} x {ny} shaped, not {s}."
            raise ValueError(msg)
        if (s := yy.shape) != (nx, ny):
            msg = f"y-locations must be {nx} x {ny} shaped, not {s}."
            raise ValueError(msg)

    def localize(  # noqa: D102
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TaylorSeriesTimeSupport:
        self._check_locations(xx, yy)
        space_fields = self._build_space()

        return TaylorSeriesTimeSupport(self._time, space_fields)

    localize.__doc__ = SpaceTimeDecomposition[
        FullFieldSpaceSupport, TaylorSeriesTimeSupport
    ].localize.__doc__
