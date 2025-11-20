"""Wavelets implementation."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from qgsw.decomposition.wavelets.basis_functions import CosineBasisFunctions
from qgsw.decomposition.wavelets.supports import (
    GaussianSupport,
    NormalizedGaussianSupport,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw import specs
from qgsw.specs import defaults

WVFunc = Callable[[torch.Tensor], torch.Tensor]


def generate_space_params(
    order: int,
    xx: torch.Tensor,
    yy: torch.Tensor,
    *,
    Lx_max: float | None = None,
    Ly_max: float | None = None,
    sigma_ratio: float | None = None,
) -> dict[str, Any]:
    """Generate space parameters for the Wavelet basis.

    Args:
        order (int): Order of decomposition.
        xx (torch.Tensor): X locations.
        yy (torch.Tensor): Y locations.
        Lx_max (float | None, optional): Largest dimension along X,
            total width if set to None. Defaults to None.
        Ly_max (float | None, optional): Largest dimension along Y,
            total width if set to None. Defaults to None.
        sigma_ratio (float | None, optional): Ratio to use to compute sigma,
            if None, set to 1/sqrt(log(2)). Defaults to None.

    Returns:
        dict[str, Any]: Space basis dictionnary.
    """
    basis = {}
    lx = (xx[-1, 0] - xx[0, 0]).cpu().item()
    ly = (yy[0, -1] - yy[0, 0]).cpu().item()
    Lx = lx if Lx_max is None else Lx_max
    Ly = ly if Ly_max is None else Ly_max
    tspecs = specs.from_tensor(xx)
    ratio = (
        1 / torch.sqrt(torch.log(torch.tensor(2, **tspecs))).cpu().item()
        if sigma_ratio is None
        else sigma_ratio
    )
    for p in range(order):
        Lx_p = Lx / 2**p
        Ly_p = Ly / 2**p
        kx_p = 2 * torch.pi / Lx_p
        ky_p = 2 * torch.pi / Ly_p

        lx_p = lx / 2**p
        xs = [xx[0, 0] + (2 * k + 1) / 2 * lx_p for k in range(2**p)]
        ly_p = ly / 2**p
        ys = [yy[0, 0] + (2 * k + 1) / 2 * ly_p for k in range(2**p)]

        centers = [
            (x.cpu().item(), y.cpu().item())
            for x, y in itertools.product(xs, ys)
        ]

        sigma_x = lx_p * ratio  # For the gaussian enveloppe
        sigma_y = ly_p * ratio  # For the gaussian enveloppe

        basis[p] = {
            "centers": centers,
            "kx": kx_p,
            "ky": ky_p,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "numel": len(centers),
        }

    return basis


def generate_time_params(
    order: int,
    tt: torch.Tensor,
    *,
    Lt_max: float | None = None,
    sigma_ratio: float | None = None,
) -> dict[str, Any]:
    """Generate time parameters for the Wavelet basis.

    Args:
        order (int): Order of decomposition.
        tt (torch.Tensor): Times.
        Lt_max (float | None, optional): Largest dimension along time,
            total width if set to None. Defaults to None.
        sigma_ratio (float | None, optional): Ratio to use to compute sigma,
            if None, set to 1/sqrt(log(2)). Defaults to None.

    Returns:
        dict[str, Any]: Time basis dictionnary.
    """
    basis = {}
    lt = (tt[-1] - tt[0]).cpu().item()
    Lt = lt if Lt_max is None else Lt_max
    tspecs = specs.from_tensor(tt)
    ratio = (
        1 / torch.sqrt(torch.log(torch.tensor(2, **tspecs))).cpu().item()
        if sigma_ratio is None
        else sigma_ratio
    )
    for p in range(order):
        lt_p = Lt / 2**p
        tc = [tt[0] + (2 * k + 1) / 2 * lt_p for k in range(2**p)]

        centers = [t.cpu().item() for t in tc]

        sigma_t = lt_p * ratio  # For the gaussian enveloppe

        basis[p] = {
            "centers": centers,
            "sigma_t": sigma_t,
            "numel": len(centers),
        }
    return basis


class WaveletBasis:
    """Wavelet decomposition.

    ΣΣ[E(t)/ΣE(t)]Σe(x,y)ΣΣcγ(x,y)

    E(t) = exp(-(t-tc)²/σ_t²)
    e(x,y) = E(x,y) / ΣE(x,y)
    E(x,y) = exp(-(x-xc)²/σ_x²)exp(-(y-yc)²/σ_y²)
    γ(x,y) = cos(kx x cos(θ) + ky y sin(θ) + φ)
    """

    _n_theta = 10

    @property
    def n_theta(self) -> int:
        """Number of orientations to consider."""
        return self._n_theta

    @n_theta.setter
    def n_theta(self, n_theta: int) -> None:
        self._n_theta = n_theta
        theta = torch.linspace(
            0, 2 * torch.pi, self.n_theta + 1, **self._specs
        )[:-1]
        self._cos_t = torch.cos(theta)
        self._sin_t = torch.sin(theta)

    @property
    def order(self) -> int:
        """Decomposition order."""
        return self._order

    def __init__(
        self,
        space_params: dict[int, dict[str, Any]],
        time_params: dict[int, dict[str, Any]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the Wavelet basis.

        Args:
            space_params (dict[int, dict[str, Any]]): Space parameters.
            time_params (dict[int, dict[str, Any]]): Time parameters.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Ddevice.
                Defaults to None.
        """
        self._check_validity(space_params, time_params)
        self._order = len(space_params.keys())
        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params
        self.n_theta = self._n_theta
        self.phase = torch.tensor([0, torch.pi / 2], **self._specs)

    def _check_validity(
        self,
        space_params: dict[str, Any],
        time_params: dict[str, Any],
    ) -> None:
        """Check parameters validity."""
        if space_params.keys() != time_params.keys():
            msg = "Mismatching keys between space and time parameters."
            raise ValueError(msg)

    def numel(self) -> int:
        """Total number of elements."""
        return sum((2**i) ** 3 for i in range(self._order)) * 2 * self.n_theta

    def generate_random_coefs(self) -> dict[int, torch.Tensor]:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            dict[int, torch.Tensor]: Level -> coefficients.
                ├── 0: (1, 1)-shaped
                ├── 1: (2, 4)-shaped
                ├── 2: (4, 16)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2)-shaped
        """
        coefs = {}
        for o in range(self._order):
            coefs[o] = torch.randn(
                (
                    self._time[o]["numel"],
                    self._space[o]["numel"],
                    self.n_theta,
                    2,
                ),
                **self._specs,
            )
        return coefs

    def set_coefs(self, coefs: dict[int, torch.Tensor]) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (torch.Tensor): Coefficients.
                ├── 0: (1, 1)-shaped
                ├── 1: (2, 4)-shaped
                ├── 2: (4, 16)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2)-shaped
        """
        self._coefs = coefs

    def _compute_space_params(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        centers = params["centers"]
        kx = params["kx"]
        ky = params["ky"]
        xc = torch.tensor([c[0] for c in centers], **self._specs)
        yc = torch.tensor([c[1] for c in centers], **self._specs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]
        return (x, y, kx, ky)

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build space-related fields.

        Σe(x,y)ΣΣcγ(x,y)

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            lambd = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.field)
            # ꟛ = e γ
            fields[lvl] = torch.einsum("tcop,cxyop->txyop", c, lambd).mean(
                dim=[-1, -2]
            )
        return fields

    def _build_space_dx(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build x-derivatives of space fields.

        ΣΣΣc[e'(x,y)γ(x,y)+e'(x,y)γ'(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dx, gamma.field)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dx)
            # coefs * (e' γ + e γ')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build second order x-derivatives of space fields.

        ΣΣΣc[e''(x,y)γ(x,y) + 2e(x,y)'γ'(x,y) + e(x,y)γ''(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e'' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dx2, gamma.field)
            # ꟛ2 = 2e' γ'
            lambda2 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dx, gamma.dx)
            # ꟛ3 = e γ''
            lambda3 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dx2)
            # coefs * (e'' γ +2e' γ'+ e γ'')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2 + lambda3
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build third order x-derivatives of space fields.

        ΣΣΣc[e'''γ + 3e''γ'+ 3e'γ'' +eγ''']

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e''' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dx3, gamma.field)
            # ꟛ2 = 3e'' γ'
            lambda2 = 3 * torch.einsum("cxy,cxyop->cxyop", e.dx2, gamma.dx)
            # ꟛ3 = 3e' γ''
            lambda3 = 3 * torch.einsum("cxy,cxyop->cxyop", e.dx, gamma.dx2)
            # ꟛ4 = e γ'''
            lambda4 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dx3)
            # coefs * (e''' γ + 3e' γ'' + 3 e' γ'' + e γ''')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                lambda1 + lambda2 + lambda3 + lambda4,
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dydx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build the x-x-y derivative of space fields.

        ΣΣΣc[e_xxy γ + e_xx γ_y + 2e_xy γ + 2e_x γ_y + e_y γ_xx + e γ_xxy]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = ∂_xxy e γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dydx2, gamma.field)
            # ꟛ2 = ∂_xx e ∂_y γ
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e.dx2, gamma.dy)
            # ꟛ3 = 2 ∂_xy e ∂_x γ
            lambda3 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dydx, gamma.dx)
            # ꟛ4 = 2 ∂_x e ∂_xy γ
            lambda4 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dx, gamma.dydx)
            # ꟛ5 = ∂_y e ∂_xx γ
            lambda5 = torch.einsum("cxy,cxyop->cxyop", e.dy, gamma.dx2)
            # ꟛ5 = e ∂_xxy γ
            lambda6 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dydx2)
            # coefs * ( ∂_xxy e γ + ∂_xx e ∂_y γ +
            #   2 ∂_xy e ∂_x γ + 2 ∂_x e ∂_xy γ +
            #   ∂_y e ∂_xx γ + e ∂_xxy γ )
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                lambda1 + lambda2 + lambda3 + lambda4 + lambda5 + lambda6,
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build t-derivatives of space fields.

        ΣΣΣc[e'(x,y)γ(x,y)+e'(x,y)γ'(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dy, gamma.field)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dy)
            # coefs * (e' γ + e γ')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build second order y-derivatives of space fields.

        ΣΣΣc[e''(x,y)γ(x,y) + 2e(x,y)'γ'(x,y) + e(x,y)γ''(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e'' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dy2, gamma.field)
            # ꟛ2 = 2e' γ'
            lambda2 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dy, gamma.dy)
            # ꟛ3 = e γ''
            lambda3 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dy2)
            # coefs * (e'' γ +2e' γ'+ e γ'')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2 + lambda3
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build third order y-derivatives of space fields.

        ΣΣΣc[e'''γ + 3e''γ'+ 3e'γ'' +eγ''']

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e''' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dy3, gamma.field)
            # ꟛ2 = 3e'' γ'
            lambda2 = 3 * torch.einsum("cxy,cxyop->cxyop", e.dy2, gamma.dy)
            # ꟛ3 = 3e' γ''
            lambda3 = 3 * torch.einsum("cxy,cxyop->cxyop", e.dy, gamma.dy2)
            # ꟛ4 = e γ'''
            lambda4 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dy3)
            # coefs * (e''' γ + 3e' γ'' + 3 e' γ'' + e γ''')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                lambda1 + lambda2 + lambda3 + lambda4,
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dxdy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build the y-y-x derivative of space fields.

        ΣΣΣc[e_yyx γ + e_yy γ_x + 2e_yx γ + 2e_y γ_x + e_x γ_yy + e γ_yyx]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = ∂_yyx e γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", e.dxdy2, gamma.field)
            # ꟛ2 = ∂_yy e ∂_x γ
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e.dy2, gamma.dx)
            # ꟛ3 = 2 ∂_yx e ∂_y γ
            lambda3 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dxdy, gamma.dy)
            # ꟛ4 = 2 ∂_y e ∂_yx γ
            lambda4 = 2 * torch.einsum("cxy,cxyop->cxyop", e.dy, gamma.dxdy)
            # ꟛ5 = ∂_x e ∂_yy γ
            lambda5 = torch.einsum("cxy,cxyop->cxyop", e.dx, gamma.dy2)
            # ꟛ6 = e ∂_yyx γ
            lambda6 = torch.einsum("cxy,cxyop->cxyop", e.field, gamma.dxdy2)
            # coefs * ( ∂_yyx e γ + ∂_yy e ∂_x γ +
            #   2 ∂_yx e ∂_y γ + 2 ∂_y e ∂_yx γ +
            #   ∂_x e ∂_yy γ + e ∂_yyx γ )
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                lambda1 + lambda2 + lambda3 + lambda4 + lambda5 + lambda6,
            ).mean(dim=[-1, -2])

        return fields

    @staticmethod
    def _at_time(
        t: torch.Tensor,
        space_fields: dict[int, torch.Tensor],
        time_params: dict[int, dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the total field value at a given time.

        Args:
            t (torch.Tensor): Time to compute field at.
            space_fields (dict[int, torch.Tensor]): Space-only fields.
            time_params (dict[int, dict[str, Any]]): Time parameters.

        Returns:
            torch.Tensor: Resulting field.
        """
        field = torch.zeros_like(space_fields[0][0].detach())
        tspecs = specs.from_tensor(t)
        for lvl, params in time_params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_ = exp / exp.sum(dim=0)

            field_at_lvl = torch.einsum("t,txy->xy", exp_, space_fields[lvl])

            field += field_at_lvl
        return field / len(time_params)

    @staticmethod
    def _dt_at_time(
        t: torch.Tensor,
        space_fields: torch.Tensor,
        time_params: dict[int, dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the total time-derivated field value at a given time.

        Args:
            t (torch.Tensor): Time to compute field at.
            space_fields (dict[int, torch.Tensor]): Space-only fields.
            time_params (dict[int, dict[str, Any]]): Time parameters.

        Returns:
            torch.Tensor: Resulting field.
        """
        field = torch.zeros_like(space_fields[0][0].detach())
        tspecs = specs.from_tensor(t)
        for lvl, params in time_params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]
            space = space_fields[lvl]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_s = exp.sum(dim=0)
            dt_exp = -2 * (t - tc) / st**2 * exp

            dt_e = (dt_exp * exp_s - exp * dt_exp.sum(dim=0)) / exp_s**2

            field_at_lvl = torch.einsum("t,txy->xy", dt_e, space)

            field += field_at_lvl
        return field / len(time_params)

    def localize(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dt(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets time derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._dt_at_time(t, space_fields, self._time)

        return at_time

    def localize_dx(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dx2(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets second order x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx2(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dx3(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets third order x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx3(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dydx2(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets x-x-y derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dydx2(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dy(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dy(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dy2(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets second order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dy2(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dy3(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets third order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dy3(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dxdy2(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets y-y-x derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dxdy2(xx=xx, yy=yy)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_laplacian(self, xx: torch.Tensor, yy: torch.Tensor) -> WVFunc:
        """Localize wavelets second order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        dx2 = self._build_space_dx2(xx=xx, yy=yy)
        dy2 = self._build_space_dy2(xx=xx, yy=yy)
        space_fields = {k: dx2[k] + dy2[k] for k in dx2}

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dx_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> WVFunc:
        """Localize wavelets x derivative of laplacian.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        dx3 = self._build_space_dx3(xx=xx, yy=yy)
        dxdy2 = self._build_space_dxdy2(xx=xx, yy=yy)
        space_fields = {k: dx3[k] + dxdy2[k] for k in dx3}

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dy_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> WVFunc:
        """Localize wavelets x derivative of laplacian.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        dydx2 = self._build_space_dydx2(xx=xx, yy=yy)
        dy3 = self._build_space_dy3(xx=xx, yy=yy)
        space_fields = {k: dydx2[k] + dy3[k] for k in dydx2}

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._at_time(t, space_fields, self._time)

        return at_time

    def localize_dt_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> WVFunc:
        """Localize wavelets second order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        dx2 = self._build_space_dx2(xx=xx, yy=yy)
        dy2 = self._build_space_dy2(xx=xx, yy=yy)
        space_fields = {k: dx2[k] + dy2[k] for k in dx2}

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis._dt_at_time(t, space_fields, self._time)

        return at_time

    @classmethod
    def from_xyt(
        cls,
        xx: torch.Tensor,
        yy: torch.Tensor,
        tt: torch.Tensor,
        *,
        order: int = 4,
        Lx_max: float | None = None,
        Ly_max: float | None = None,
        Lt_max: float | None = None,
        sigma_ratio: float | None = None,
    ) -> Self:
        """Instantiate the WaveletBasis from x,y and t.

        Args:
            xx (torch.Tensor): Xs.
            yy (torch.Tensor): Ys.
            tt (torch.Tensor): Times.
            order (int, optional): Decomposition order. Defaults to 4.
            Lx_max (float | None, optional): Largest dimension along X,
                total width if set to None. Defaults to None.
            Ly_max (float | None, optional): Largest dimension along Y,
                total width if set to None. Defaults to None.
            Lt_max (float | None, optional): Largest dimension along time,
                total width if set to None. Defaults to None.
            sigma_ratio (float | None, optional): Ratio to use to compute
                sigma, if None, set to 1/sqrt(log(2)). Defaults to None.

        Returns:
            Self: WaveletBasis.
        """
        space_params = generate_space_params(
            order=order,
            xx=xx,
            yy=yy,
            Lx_max=Lx_max,
            Ly_max=Ly_max,
            sigma_ratio=sigma_ratio,
        )
        time_params = generate_time_params(
            order=order, tt=tt, Lt_max=Lt_max, sigma_ratio=sigma_ratio
        )

        return cls(space_params, time_params)
