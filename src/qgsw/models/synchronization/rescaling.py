"""Rescaling tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.prognostic_tuples import (
    UVH,
    UVHT,
    UVHTAlpha,
)
from qgsw.masks import Masks
from qgsw.models.names import ModelCategory
from qgsw.specs import defaults

if TYPE_CHECKING:
    from qgsw.fields.variables.state import StateUVH, StateUVHAlpha
    from qgsw.models.base import ModelUVH
    from qgsw.models.qg.uvh.projectors.core import QGProjector


def interpolate_physical_variable(
    tensor: torch.Tensor,
    size_out: tuple[int, int],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interpolate physical variable.

    Args:
        tensor (torch.Tensor): Variable tensor.
        size_out (tuple[int, int]): Output horizontal size.
        mask (torch.Tensor | None, optional): Output mask. Defaults to None.

    Returns:
        torch.Tensor: Interpolated variable.
    """
    tensor_i = F.interpolate(
        tensor,
        size=size_out,
        mode="bicubic",
        align_corners=False,
    )
    if mask is not None:
        return tensor_i * mask
    return tensor_i


class Rescaler:
    """Rescaler."""

    __slots__ = ("_P", "_dx", "_dy", "_masks", "_nx", "_ny")

    def __init__(
        self,
        nx_out: int,
        ny_out: int,
        dx_out: float,
        dy_out: float,
        masks: Masks | None = None,
        qg_proj: QGProjector | None = None,
    ) -> None:
        """Instantiate Rescaler..

        Args:
            nx_out (int): Output nx.
            ny_out (int): Output ny.
            dx_out (float): Output dx.
            dy_out (float): Output dy.
            masks (Masks | None, optional): Masks for output. Defaults to None.
            qg_proj (QGProjector | None, optional): QG Projector, if None the
                rescale uvh is not guaranteed to be quai-geostrophic.
                Defaults to None.
        """
        self._nx = nx_out
        self._ny = ny_out
        self._dx = dx_out
        self._dy = dy_out
        self._masks = masks or Masks.empty(
            nx_out,
            ny_out,
            device=defaults.get_device(),
        )
        self._P = qg_proj

    def _raise_if_incompatible_shapes(
        self,
        *,
        nxin: int,
        nyin: int,
    ) -> None:
        """Raise an error if the model have incompatible shapes.

        Args:
            nxin (int): Input nx.
            nyin (int): Input ny.

        Raises:
            ValueError: If nx (or, resp. nxin) does not divide nxin
                (or, resp. nx).
            ValueError: If ny (or, resp. nyin) does not divide nyin
                (or, resp. ny).
            ValueError: If nx / nxin != ny / nyin
        """
        if (nxin % self._nx != 0) and (self._nx % nxin != 0):
            msg = (
                "nxin (or, resp. self._nx) must divisable"
                " by self._nx (or, resp. nxin)."
            )
            raise ValueError(msg)
        if (nyin % self._ny != 0) and (self._ny % nyin != 0):
            msg = (
                "nyin (or, resp. self._ny) must divisable"
                " by self._ny (or, resp. nyin)."
            )
            raise ValueError(msg)
        if self._nx / nxin != self._ny / nyin:
            msg = (
                "There should be the same ratio between"
                " self._nx / nxin and self._ny / nyin"
            )
            raise ValueError(msg)

    def _to_physical(self, uvh: UVH, dx: float, dy: float) -> UVH:
        """Convert physical uvh to covariant.

        Args:
            uvh (UVH): Covariant uvh: u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            dx (float): Infinitesimal distance in the X direction.
            dy (float): Infinitesimal distance in the Y direction.

        Returns:
            UVH: Physical uvh: u_phys,v_phys and h_phys.
                ├── u_phys = u / dx: (n_ens, nl, nx+1, ny)-shaped
                ├── v_phys = v / dy: (n_ens, nl, nx, ny+1)-shaped
                └── h_phys = h / (dx * dy): (n_ens, nl, nx, ny)-shaped
        """
        return UVH(uvh.u / dx, uvh.v / dy, uvh.h / (dx * dy))

    def _to_covariant(self, uvh: UVH, dx: float, dy: float) -> UVH:
        """Convert physical uvh to covariant.

        Args:
            uvh (UVH): Physical uvh: u_phys,v_phys and h_phys.
                ├── u_phys: (n_ens, nl, nx+1, ny)-shaped
                ├── v_phys: (n_ens, nl, nx, ny+1)-shaped
                └── h_phys: (n_ens, nl, nx, ny)-shaped
            dx (float): Infinitesimal distance in the X direction.
            dy (float): Infinitesimal distance in the Y direction.

        Returns:
            UVH: Covariant uvh: u,v and h.
                ├── u = u_phys * dx: (n_ens, nl, nx+1, ny)-shaped
                ├── v = v_phys * dy: (n_ens, nl, nx, ny+1)-shaped
                └── h = h_phys * (dx * dy): (n_ens, nl, nx, ny)-shaped
        """
        return UVH(uvh.u * dx, uvh.v * dy, uvh.h * (dx * dy))

    def _ensure_mass_conservation(
        self,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        hmask: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce mass conservation.

        Args:
            h_in (torch.Tensor): Input h.
            h_out (torch.Tensor): Output h.
            hmask (torch.Tensor): H mask.

        Returns:
            torch.Tensor: Output h with conserved mass.
        """
        for k in range(h_in.shape[-3]):
            h_out[0, k] += h_in[0, k].mean() - h_out[0, k].mean()
        return h_out * hmask

    def rescale_uvh(self, uvh: UVH, dx_in: float, dy_in: float) -> UVH:
        """Rescale uvh.

        Args:
            uvh (UVH): Input uvh: u,v and h.
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            dx_in (float): Input dx.
            dy_in (float): Input dy.

        Returns:
            UVH: Output uvh: u,v and h.
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        u, v, h = self._to_physical(uvh, dx_in, dy_in)
        _, _, nx_in, ny_in = h.shape
        self._raise_if_incompatible_shapes(nxin=nx_in, nyin=ny_in)

        # INTERPOLATION ------------------------------------------------------
        ## Physical zonal velocity
        usize_out = (self._nx + 1, self._ny)
        umask = self._masks.u
        u_i = interpolate_physical_variable(u, usize_out, umask)
        ## Physical meriodional velocity
        vsize_out = (self._nx, self._ny + 1)
        vmask = self._masks.v
        v_i = interpolate_physical_variable(v, vsize_out, vmask)
        ## Physical layer depth anomaly
        hsize_out = (self._nx, self._ny)
        hmask = self._masks.h
        h_i = interpolate_physical_variable(h, hsize_out, hmask)
        # --------------------------------------------------------------------

        uvh_i = UVH(u_i, v_i, h_i)  # Physical

        if self._P:
            uvh_i_cov = self._to_covariant(
                UVH(u_i, v_i, h_i),
                self._dx,
                self._dy,
            )  # Covariant
            uvh_i_cov = self._P.project(uvh_i_cov)
            uvh_i = self._to_physical(
                uvh_i_cov,
                self._dx,
                self._dy,
            )  # Physical

        h_i = self._ensure_mass_conservation(h, uvh_i.h, hmask=hmask)

        return self._to_covariant(
            UVH(uvh_i.u, uvh_i.v, h_i),
            self._dx,
            self._dy,
        )  # Covariant

    def rescale_uvht(self, uvht: UVHT, dx_in: float, dy_in: float) -> UVHT:
        """Rescale UVHT.

        Args:
            uvht (UVHT): Input uvht: t,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            dx_in (float): Input dx.
            dy_in (float): Input dy.

        Returns:
            UVHT: Output uvht: t,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        return UVHT.from_uvh(
            uvht.t,
            self.rescale_uvh(uvht.uvh, dx_in, dy_in),
        )

    def rescale_uvht_alpha(
        self,
        uvht_alpha: UVHTAlpha,
        dx_in: float,
        dy_in: float,
    ) -> UVHTAlpha:
        """Rescale UVHTAlpha.

        Args:
            uvht_alpha (UVHTAlpha): Input uvhtα: t,α,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── α: (n_ens, 1, nx_in, ny_in)-shaped
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            dx_in (float): Input dx.
            dy_in (float): Input dy.

        Returns:
            UVHTAlpha: Output uvhtα: t,α,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── α: (n_ens, 1, nx_out, ny_out)-shaped
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        alpha = uvht_alpha.alpha
        alpha_i = interpolate_physical_variable(
            alpha,
            size_out=(self._nx, self._ny),
        )
        return UVHTAlpha.from_uvh(
            uvht_alpha.t,
            alpha_i,
            self.rescale_uvh(uvht_alpha.uvh, dx_in, dy_in),
        )

    @overload
    def __call__(self, prognostic: UVH, dx_in: float, dy_in: float) -> UVH: ...
    @overload
    def __call__(
        self,
        prognostic: UVHT,
        dx_in: float,
        dy_in: float,
    ) -> UVHT: ...
    @overload
    def __call__(
        self,
        prognostic: UVHTAlpha,
        dx_in: float,
        dy_in: float,
    ) -> UVHTAlpha: ...
    def __call__(
        self,
        prognostic: UVH | UVHT | UVHTAlpha,
        dx_in: float,
        dy_in: float,
    ) -> UVH | UVHT | UVHTAlpha:
        """Perform rescaling.

        Args:
            prognostic (UVH | UVHT | UVHTAlpha): (t,α,)u,v and h.
                ├── (t: (n_ens, )-shaped)
                ├── (α: (n_ens, 1, nx_in, ny_in)-shaped)
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            dx_in (float): Input dx.
            dy_in (float): Input dy.

        Raises:
            TypeError: If prognostic is neither UVH, UVHT or UVHTAlpha

        Returns:
            UVH | UVHT | UVHTAlpha: (t,α,)u,v and h.
                ├── (t: (n_ens, )-shaped)
                ├── (α: (n_ens, 1, nx_out, ny_out)-shaped)
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        if isinstance(prognostic, UVH):
            return self.rescale_uvh(prognostic, dx_in, dy_in)
        if isinstance(prognostic, UVHT):
            return self.rescale_uvht(prognostic, dx_in, dy_in)
        if isinstance(prognostic, UVHTAlpha):
            return self.rescale_uvht_alpha(prognostic, dx_in, dy_in)
        msg = "Unsupported prognostic type."
        raise TypeError(msg)

    @overload
    def from_model(self, model: ModelUVH[UVHT, StateUVH]) -> UVHT: ...
    @overload
    def from_model(
        self,
        model: ModelUVH[UVHTAlpha, StateUVHAlpha],
    ) -> UVHTAlpha: ...
    def from_model(self, model: ModelUVH) -> UVHT | UVHTAlpha:
        """Use model to infer input uvh.

        Args:
            model (ModelUVH): Model to infer uvh from.

        Returns:
            UVHT | UVHTAlpha: Interpolated prognostic.
        """
        is_qg = model.get_category() == ModelCategory.QUASI_GEOSTROPHIC
        qg_proj = model.P if is_qg else None
        return self(
            model.prognostic,
            model.space.dx,
            model.space.dy,
            model.masks,
            qg_proj,
        )

    @classmethod
    def for_model(cls, model: ModelUVH) -> Rescaler:
        """Set the rescaler for a given model.

        Args:
            model (ModelUVH): Model to use as Rescaler output.

        Returns:
            Rescaler: Rescaler.
        """
        qg_proj = (
            model.P
            if model.get_category() == ModelCategory.QUASI_GEOSTROPHIC
            else None
        )
        return cls(
            model.space.nx,
            model.space.ny,
            model.space.dx,
            model.space.dy,
            model.masks,
            qg_proj,
        )
