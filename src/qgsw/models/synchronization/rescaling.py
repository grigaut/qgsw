"""Rescaling tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.exceptions import RescalingShapeMismatchError
from qgsw.fields.variables.tuples import (
    UVH,
    UVHT,
    UVHTAlpha,
)
from qgsw.logging import getLogger
from qgsw.masks import Masks
from qgsw.models.names import ModelCategory
from qgsw.specs import defaults
from qgsw.utils import covphys

if TYPE_CHECKING:
    from qgsw.fields.variables.state import StateUVH, StateUVHAlpha
    from qgsw.models.base import ModelUVH
    from qgsw.models.qg.uvh.projectors.core import QGProjector


logger = getLogger(__name__)


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

    __slots__ = ("_dx", "_dy", "_masks", "_nx", "_ny")

    def __init__(
        self,
        nx_out: int,
        ny_out: int,
        dx_out: float,
        dy_out: float,
        masks: Masks | None = None,
    ) -> None:
        """Instantiate Rescaler..

        Args:
            nx_out (int): Output nx.
            ny_out (int): Output ny.
            dx_out (float): Output dx.
            dy_out (float): Output dy.
            masks (Masks | None, optional): Masks for output. Defaults to None.
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

    @property
    def output_hshape(self) -> tuple[int, int]:
        """Output horizontal shape."""
        return (self._nx, self._ny)

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
            RescalingShapeMismatchError: If nx (or, resp. nxin)
                does not divide nxin (or, resp. nx).
            RescalingShapeMismatchError: If ny (or, resp. nyin)
                does not divide nyin (or, resp. ny).
            RescalingShapeMismatchError: If nx / nxin != ny / nyin
        """
        if (nxin % self._nx != 0) and (self._nx % nxin != 0):
            msg = (
                "nxin (or, resp. self._nx) must divisable"
                " by self._nx (or, resp. nxin)."
            )
            raise RescalingShapeMismatchError(msg)
        if (nyin % self._ny != 0) and (self._ny % nyin != 0):
            msg = (
                "nyin (or, resp. self._ny) must divisable"
                " by self._ny (or, resp. nyin)."
            )
            raise RescalingShapeMismatchError(msg)
        if self._nx / nxin != self._ny / nyin:
            msg = (
                "There should be the same ratio between"
                " self._nx / nxin and self._ny / nyin"
            )
            raise RescalingShapeMismatchError(msg)

    def _ensure_mass_conservation(
        self,
        h_in: torch.Tensor,
        h_out: torch.Tensor,
        hmask: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce mass conservation.

        Args:
            h_in (torch.Tensor): Input (physical) h.
            h_out (torch.Tensor): Interpolated input (physical) h.
            hmask (torch.Tensor): H mask.

        Returns:
            torch.Tensor: Interpolated (physical) h with conserved mass.
        """
        for k in range(h_in.shape[-3]):
            h_out[0, k] += h_in[0, k].mean() - h_out[0, k].mean()
        return h_out * hmask

    def rescale_uvh(self, uvh: UVH, proj: QGProjector | None = None) -> UVH:
        """Rescale uvh.

        WARNING: Rescaling is performed on physical variables.

        Rescaling process:
        1. Interpolate u, v and h.
        2. If the output is expected to quasi-geostrophic: Project u,v and h
        3. Ensure mass conservation by corrected h amplitudes

        Args:
            uvh (UVH): Input (physical) uvh: u,v and h.
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            proj (QGProjector | None, optional): QG projector related with
                input data. If None, the rescaled data won't be QG.
                Defaults to None.

        Returns:
            UVH: Output (physical) uvh: u,v and h.
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        u, v, h = uvh
        _, _, nx_in, ny_in = h.shape
        self._raise_if_incompatible_shapes(nxin=nx_in, nyin=ny_in)

        msg = (
            f"Rescaling from nx={nx_in}, ny={ny_in} to "
            f"nx={self._nx}, ny={self._ny}."
        )
        logger.detail(msg)

        # INTERPOLATION ------------------------------------------------------
        ## Physical zonal velocity
        usize_out = (self._nx + 1, self._ny)
        umask = self._masks.u
        u_i = interpolate_physical_variable(u, usize_out, umask)
        ## Physical meridional velocity
        vsize_out = (self._nx, self._ny + 1)
        vmask = self._masks.v
        v_i = interpolate_physical_variable(v, vsize_out, vmask)
        ## Physical layer depth anomaly
        hsize_out = (self._nx, self._ny)
        hmask = self._masks.h
        h_i = interpolate_physical_variable(h, hsize_out, hmask)
        # --------------------------------------------------------------------

        uvh_i = UVH(u_i, v_i, h_i)  # Physical

        if proj is not None:
            proj_i = proj.to_shape(self._nx, self._ny)
            uvh_i_cov = covphys.to_cov(
                UVH(u_i, v_i, h_i),
                self._dx,
                self._dy,
            )  # Covariant
            uvh_i_cov = proj_i.project(
                uvh_i_cov,
            )
            uvh_i = covphys.to_phys(
                uvh_i_cov,
                self._dx,
                self._dy,
            )  # Physical

        h_i = self._ensure_mass_conservation(h, uvh_i.h, hmask=hmask)

        return UVH(uvh_i.u, uvh_i.v, h_i)

    def rescale_uvht(
        self,
        uvht: UVHT,
        proj: QGProjector | None = None,
    ) -> UVHT:
        """Rescale UVHT.

        Args:
            uvht (UVHT): Input (physical) uvht: t,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            proj (QGProjector | None, optional): QG projector related with
                input data. If None, the rescaled data won't be QG.
                Defaults to None.

        Returns:
            UVHT: Output (physical) uvht: t,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        return UVHT.from_uvh(
            uvht.t,
            self.rescale_uvh(uvht.uvh, proj),
        )

    def rescale_uvht_alpha(
        self,
        uvht_alpha: UVHTAlpha,
        proj: QGProjector | None = None,
    ) -> UVHTAlpha:
        """Rescale UVHTAlpha.

        Args:
            uvht_alpha (UVHTAlpha): Input (physical) uvhtα: t,α,u,v and h.
                ├── t: (n_ens, )-shaped
                ├── α: (n_ens, 1, nx_in, ny_in)-shaped
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            proj (QGProjector | None, optional): QG projector related with
                input data. If None, the rescaled data won't be QG.
                Defaults to None.

        Returns:
            UVHTAlpha: Output (physical) uvhtα: t,α,u,v and h.
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
            self.rescale_uvh(uvht_alpha.uvh, proj),
        )

    @overload
    def __call__(
        self,
        prognostic: UVH,
        proj: QGProjector | None = None,
    ) -> UVH: ...
    @overload
    def __call__(
        self,
        prognostic: UVHT,
        proj: QGProjector | None = None,
    ) -> UVHT: ...
    @overload
    def __call__(
        self,
        prognostic: UVHTAlpha,
        proj: QGProjector | None = None,
    ) -> UVHTAlpha: ...
    def __call__(
        self,
        prognostic: UVH | UVHT | UVHTAlpha,
        proj: QGProjector | None = None,
    ) -> UVH | UVHT | UVHTAlpha:
        """Perform rescaling.

        Args:
            prognostic (UVH | UVHT | UVHTAlpha): (physical) (t,α,)u,v and h.
                ├── (t: (n_ens, )-shaped)
                ├── (α: (n_ens, 1, nx_in, ny_in)-shaped)
                ├── u: (n_ens, nl, nx_in+1, ny_in)-shaped
                ├── v: (n_ens, nl, nx_in, ny_in+1)-shaped
                └── h: (n_ens, nl, nx_in, ny_in)-shaped
            proj (QGProjector | None, optional): QG projector related with
                input data. If None, the rescaled data won't be QG.
                Defaults to None.

        Raises:
            TypeError: If prognostic is neither UVH, UVHT or UVHTAlpha

        Returns:
            UVH | UVHT | UVHTAlpha: (physical) (t,α,)u,v and h.
                ├── (t: (n_ens, )-shaped)
                ├── (α: (n_ens, 1, nx_out, ny_out)-shaped)
                ├── u: (n_ens, nl, nx_out+1, ny_out)-shaped
                ├── v: (n_ens, nl, nx_out, ny_out+1)-shaped
                └── h: (n_ens, nl, nx_out, ny_out)-shaped
        """
        if isinstance(prognostic, UVH):
            return self.rescale_uvh(prognostic, proj)
        if isinstance(prognostic, UVHT):
            return self.rescale_uvht(prognostic, proj)
        if isinstance(prognostic, UVHTAlpha):
            return self.rescale_uvht_alpha(prognostic, proj)
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
            model.physical,
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
        return cls(
            model.space.nx,
            model.space.ny,
            model.space.dx,
            model.space.dy,
            model.masks,
        )
