"""Space support function for VarDyn basis."""

from math import comb, pi, sqrt
from typing import NamedTuple

import torch

from qgsw import specs
from qgsw.decomposition.supports.space.base import SpaceSupportFunction


class SupportLayout(NamedTuple):
    """Geometry shared by space and time supports."""

    ip: torch.Tensor  # (nsupp,) wavelet index
    ix: torch.Tensor  # (nsupp,) grid point index
    np: int
    nphys: int


class VDSpaceSupport(SpaceSupportFunction):
    """Space support function for VarDyn-like basis.

    Each `_compute*` returns a sparse operator of shape
    (nsupp, NP * ntheta * 2) such that
        field = operator @ coefs.T        # (nsupp, Nt)
    with coefs of shape (Nt, NP * ntheta * 2), ordered (p, theta, cos/sin).
    """

    def __init__(
        self,
        lons: torch.Tensor,
        lats: torch.Tensor,
        lons_centers: torch.Tensor,
        lats_centers: torch.Tensor,
        k: torch.Tensor,
        Lx: float,
        thetas: torch.Tensor,
    ) -> None:
        """Instantiate the Space support function.

        Args:
            lons (torch.Tensor): Grid longitudes coordinates (rad).
            lats (torch.Tensor): Grid latitudes coordinates (rad).
            lons_centers (torch.Tensor): Wavelet latitudes centers (rad).
            lats_centers (torch.Tensor): Wavelet longitudes centers (rad).
            k (torch.Tensor): Wavenumber (rad/km).
            Lx (float): Horizontal scale (km).
            thetas (torch.Tensor): Orientations (ntheta,).
        """
        self.device = specs.from_tensor(lons)["device"]
        rad_per_m = pi / 180 / 110 / 1000

        self.ntheta = thetas.shape[0]
        self.np = lons_centers.numel()
        self.nphys = lons.numel()
        self.Lx = Lx

        lons_ = lons.flatten()[None, :]
        lats_ = lats.flatten()[None, :]
        lons_c = lons_centers[:, None]
        lats_c = lats_centers[:, None]
        coslat0 = torch.cos(lats_c)

        # Support (chunked over wavelets)
        xs = (lons_ - lons_c) / rad_per_m * coslat0
        ys = (lats_ - lats_c) / rad_per_m
        mask = (xs.abs() <= self.Lx) & (ys.abs() <= self.Lx)
        self.ip, self.ix = mask.nonzero(as_tuple=True)
        self.xs = xs[self.ip, self.ix]
        self.ys = ys[self.ip, self.ix]
        self.nsupp = self.ip.numel()

        self.kx = (k * torch.cos(thetas))[:, None]  # (ntheta, 1)
        self.ky = (k * torch.sin(thetas))[:, None]

        # COO indices of the transposed operator, built (and sorted) once
        m = torch.arange(self.ntheta, device=self.device)[:, None, None]
        c = torch.arange(2, device=self.device)[None, None, :]
        rows = (self.ip[None, :, None] * self.ntheta + m) * 2 + c
        cols = torch.arange(self.nsupp, device=self.device)[None, :, None]
        cols = cols.expand_as(rows)
        ncols = self.np * self.ntheta * 2
        self._perm = torch.argsort((cols * ncols + rows).reshape(-1))
        self._indices = torch.stack([cols, rows]).reshape(2, -1)[:, self._perm]
        self.size = (self.nsupp, ncols)

    @property
    def layout(self) -> SupportLayout:
        """Layout."""
        return SupportLayout(self.ip, self.ix, self.np, self.nphys)

    # ---- building blocks -------------------------------------------------
    def _window_deriv(self, u: torch.Tensor, j: int) -> torch.Tensor:
        """j-th derivative of cos²(a u)."""
        a = pi / (2 * self.Lx)
        if j == 0:
            return torch.cos(a * u).square()
        return 0.5 * (2 * a) ** j * torch.cos(2 * a * u + j * pi / 2)

    def _axis(self, u: torch.Tensor, k: torch.Tensor, n: int) -> torch.Tensor:
        """n-th derivative of w(u) e^{i k u}, complex (ntheta, nsupp)."""
        phase = k * u[None, :]
        e = torch.polar(torch.ones_like(phase), phase)
        out = 0
        for j in range(n + 1):
            out = (
                out
                + comb(n, j)
                * self._window_deriv(u, j)[None, :]
                * ((1j * k) ** (n - j))
                * e
            )
        return out

    def _operator(self, nx: int = 0, ny: int = 0) -> torch.Tensor:
        """Operator of the ∂x^nx ∂y^ny derivative (per meter)."""
        z = (
            sqrt(2)
            * self._axis(self.xs, self.kx, nx)
            * self._axis(self.ys, self.ky, ny)
        )
        values = torch.view_as_real(z).reshape(-1)[
            self._perm
        ]  # (re=cos, im=sin)
        return torch.sparse_coo_tensor(
            self._indices, values, size=self.size, is_coalesced=True
        )

    def project(
        self, coefs: torch.Tensor, operator: torch.Tensor
    ) -> torch.Tensor:
        """(Nt, NP*ntheta*2) -> (nsupp, Nt), differentiable w.r.t. coefs."""
        return torch.sparse.mm(operator, coefs.T)

    # ---- public derivatives ---------------------------------------------
    def _compute(self) -> torch.Tensor:
        return self._operator(0, 0)

    def _compute_dx(self) -> torch.Tensor:
        return self._operator(1, 0)

    def _compute_dy(self) -> torch.Tensor:
        return self._operator(0, 1)

    def _compute_dx2(self) -> torch.Tensor:
        return self._operator(2, 0)

    def _compute_dy2(self) -> torch.Tensor:
        return self._operator(0, 2)

    def _compute_dx3(self) -> torch.Tensor:
        return self._operator(3, 0)

    def _compute_dy3(self) -> torch.Tensor:
        return self._operator(0, 3)

    def _compute_dydx2(self) -> torch.Tensor:
        return self._operator(2, 1)  # ∂y∂x²

    def _compute_dxdy2(self) -> torch.Tensor:
        return self._operator(1, 2)  # ∂x∂y²
