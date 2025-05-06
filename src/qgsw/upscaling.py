"""Data upscaling."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.prognostic_tuples import UVH, UVHT, UVHTAlpha

if TYPE_CHECKING:
    from qgsw.masks import Masks


class Upscaler:
    """Data upscaling class."""

    __slots__ = ()
    _mode = "bicubic"
    _align_corners = False

    def _upscale_uvh(self, uvh: UVH, delta: int, masks: Masks) -> UVH:
        """Upscale uvh data.

        Args:
            uvh (UVH): (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.
            masks: Masks: Masks for u, v and h.

        Returns:
            UVH: (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        u, v, h = uvh
        _, nl, nx, ny = h.shape
        u_up: torch.Tensor = (
            F.interpolate(
                u,
                size=(delta * nx + 1, delta * ny),
                mode=self._mode,
                align_corners=self._align_corners,
            )
            * masks.u
        )
        v_up: torch.Tensor = (
            F.interpolate(
                v,
                size=(delta * nx, delta * ny + 1),
                mode=self._mode,
                align_corners=self._align_corners,
            )
            * masks.v
        )
        h_up: torch.Tensor = (
            F.interpolate(
                h,
                size=(delta * nx, delta * ny),
                mode=self._mode,
                align_corners=self._align_corners,
            )
            * masks.h
        )
        return UVH(
            u=u_up,
            v=v_up,
            h=h_up,
        )

    def _upscale_uvht(
        self,
        uvht: UVHT,
        delta: int,
        masks: Masks,
    ) -> UVHT:
        """Upscale uvht data.

        Args:
            uvht (UVHT): t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            delta (int): Upscaling factor.
            masks: Masks: Masks for u, v and h.

        Returns:
            UVHTAlpha: t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── u: (n_ens, nl, δ*nx+1, δ*ny)-shaped
                ├── v: (n_ens, nl, δ*nx, δ*ny+1)-shaped
                └── h: (n_ens, nl, δ*nx, δ*ny)-shaped
        """
        return UVHT.from_uvh(
            uvht.t,
            self._upscale_uvh(uvht.uvh, delta, masks),
        )

    def _upscale_uvht_alpha(
        self,
        uvht_alpha: UVHTAlpha,
        delta: int,
        masks: Masks,
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
            masks: Masks: Masks for u, v and h.

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
            self._upscale_uvh(uvht_alpha.uvh, delta, masks),
        )

    @overload
    def __call__(self, prognostic: UVH, delta: int, masks: Masks) -> UVH: ...
    @overload
    def __call__(self, prognostic: UVHT, delta: int, masks: Masks) -> UVHT: ...
    @overload
    def __call__(
        self,
        prognostic: UVHTAlpha,
        delta: int,
        masks: Masks,
    ) -> UVHTAlpha: ...

    def __call__(
        self,
        prognostic: UVH | UVHT | UVHTAlpha,
        delta: int,
        masks: Masks,
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
            masks: Masks: Masks for u, v and h.

        Returns:
            UVH | UVHT | UVHTAlpha: (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens, 1, δ*nx, δ*ny)-shaped)
                ├── u: (n_ens, nl, δ*nx+1, δ*ny)-shaped
                ├── v: (n_ens, nl, δ*nx, δ*ny+1)-shaped
                └── h: (n_ens, nl, δ*nx, δ*ny)-shaped
        """
        if isinstance(prognostic, UVH):
            return self._upscale_uvh(prognostic, delta, masks)
        if isinstance(prognostic, UVHT):
            return self._upscale_uvht(prognostic, delta, masks)
        if isinstance(prognostic, UVHTAlpha):
            return self._upscale_uvht_alpha(prognostic, delta, masks)
        msg = "Unsupported prognostic type."
        raise TypeError(msg)
