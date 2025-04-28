"""Data upscaling."""

from __future__ import annotations

from typing import overload

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.prognostic_tuples import UVH, UVHT, UVHTAlpha
from qgsw.specs import defaults


class Upscaler:
    """Data upscaling class."""

    __slots__ = ()
    _mode = "bicubic"
    _align_corners = False

    def _upscale_uvh(self, uvh: UVH, delta: int) -> UVH:
        """Upscale uvh data.

        Args:
            uvh (UVH): (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.

        Returns:
            UVH: (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        _, _, h = uvh
        h_up: torch.Tensor = F.interpolate(
            h,
            scale_factor=delta,
            mode=self._mode,
            align_corners=self._align_corners,
        )
        n_ens, nl, nx, ny = h_up.shape
        uvh_up = UVH.steady(n_ens, nl, nx, ny, **defaults.get())
        return UVH(
            u=uvh_up.u,
            v=uvh_up.v,
            h=h_up,
        )

    def _upscale_uvht(
        self,
        uvht: UVHT,
        delta: int,
    ) -> UVHT:
        """Upscale uvht data.

        Args:
            uvht (UVHT): t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.

        Returns:
            UVHTAlpha: t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── u: (n_ens, nl, δ*nx+1, δ*ny)-shaped
                ├── v: (n_ens, nl, δ*nx, δ*ny+1)-shaped
                └── h: (n_ens, nl, δ*nx, δ*ny)-shaped
        """
        return UVHT.from_uvh(
            uvht.t,
            self._upscale_uvh(uvht.uvh, delta),
        )

    def _upscale_uvht_alpha(
        self,
        uvht_alpha: UVHTAlpha,
        delta: int,
    ) -> UVHTAlpha:
        """Upscale uvht_alpha data.

        Args:
            uvht_alpha (UVHTAlpha): t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── α: (n_ens, 1, nx, ny)-shaped
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.

        Returns:
            UVHTAlpha: t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── α: (n_ens, 1, δ*nx, δ*ny)-shaped
                ├── u: (n_ens, nl, δ*nx+1, δ*ny)-shaped
                ├── v: (n_ens, nl, δ*nx, δ*ny+1)-shaped
                └── h: (n_ens, nl, δ*nx, δ*ny)-shaped
        """
        alpha = uvht_alpha.alpha
        alpha_up: torch.Tensor = F.interpolate(
            alpha,
            scale_factor=delta,
            mode=self._mode,
            align_corners=self._align_corners,
        )
        return UVHTAlpha.from_uvh(
            uvht_alpha.t,
            alpha_up,
            self._upscale_uvh(uvht_alpha.uvh, delta),
        )

    @overload
    def __call__(self, prognostic: UVH, delta: int) -> UVH: ...
    @overload
    def __call__(self, prognostic: UVHT, delta: int) -> UVHT: ...
    @overload
    def __call__(self, prognostic: UVHTAlpha, delta: int) -> UVHTAlpha: ...

    def __call__(
        self,
        prognostic: UVH | UVHT | UVHTAlpha,
        delta: int,
    ) -> UVH | UVHT | UVHTAlpha:
        """Upscale prognostic data.

        Args:
            prognostic (UVH | UVHT | UVHTAlpha): (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens, 1, nx, ny)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.

        Returns:
            UVH | UVHT | UVHTAlpha: (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens, 1, δ*nx, δ*ny)-shaped)
                ├── u: (n_ens, nl, δ*nx+1, δ*ny)-shaped
                ├── v: (n_ens, nl, δ*nx, δ*ny+1)-shaped
                └── h: (n_ens, nl, δ*nx, δ*ny)-shaped
        """
        if isinstance(prognostic, UVH):
            return self._upscale_uvh(prognostic, delta)
        if isinstance(prognostic, UVHT):
            return self._upscale_uvht(prognostic, delta)
        if isinstance(prognostic, UVHTAlpha):
            return self._upscale_uvht_alpha(prognostic, delta)
        msg = "Unsupported prognostic type."
        raise TypeError(msg)
