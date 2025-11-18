"""Wavelets implementation."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw import specs
from qgsw.models.core.utils import OptimizableFunction
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
        theta = torch.linspace(0, torch.pi, self.n_theta, **self._specs)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            gamma = torch.cos(
                (kx_cos + ky_sin)[..., None]
                + self.phase[None, None, None, None, :]
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            # e = E / ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            lambd = torch.einsum("cxy,cxyop->cxyop", e, gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dx_gamma = -kx * torch.einsum(
                "o,cxyop->cxyop", self._cos_t, sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dx_E = -2 * x / sx**2 * E
            dx_E_s = dx_E.sum(dim=0)

            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dx_e = (
                torch.einsum("cxy,xy->cxy", dx_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dx_E_s)
            ) / E_s.square()
            # ꟛ1 = e' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dx_e, gamma)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e, dx_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dx_gamma = -kx * torch.einsum(
                "o,cxyop->cxyop", self._cos_t, sin_xy
            )
            dx2_gamma = -(kx**2) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t.square(), gamma
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dx_E = -2 * x / sx**2 * E
            dx_E_s = dx_E.sum(dim=0)
            dx2_E = (-2 / sx**2 + 4 * x**2 / sx**4) * E
            dx2_E_s = dx2_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dx_e = (
                torch.einsum("cxy,xy->cxy", dx_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dx_E_s)
            ) / E_s.square()
            # e'' = (E''(ΣE)² - EΣE''ΣE - 2E'ΣE'ΣE + 2E(ΣE')²)/(ΣE)³
            dx2_e = (
                torch.einsum("cxy,xy->cxy", dx2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dx2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dx_E, dx_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dx_E_s.square())
            ) / E_s.pow(3)

            # ꟛ1 = e'' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dx2_e, gamma)
            # ꟛ2 = 2e' γ'
            lambda2 = 2 * torch.einsum("cxy,cxyop->cxyop", dx_e, dx_gamma)
            # ꟛ3 = e γ''
            lambda3 = torch.einsum("cxy,cxyop->cxyop", e, dx2_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dx_gamma = -kx * torch.einsum(
                "o,cxyop->cxyop", self._cos_t, sin_xy
            )
            dx2_gamma = -(kx**2) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t.square(), gamma
            )
            dx3_gamma = (kx**3) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t.pow(3), sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dx_E = -2 * x / sx**2 * E
            dx_E_s = dx_E.sum(dim=0)
            dx2_E = (-2 / sx**2 + 4 * x**2 / sx**4) * E
            dx2_E_s = dx2_E.sum(dim=0)
            dx3_E = (12 * x / sx**4 - 8 * x**3 / sx**6) * E
            dx3_E_s = dx3_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dx_e = (
                torch.einsum("cxy,xy->cxy", dx_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dx_E_s)
            ) / E_s.square()
            # e'' = (E''(ΣE)² - EΣE''ΣE - 2E'ΣE'ΣE + 2E(ΣE')²)/(ΣE)³
            dx2_e = (
                torch.einsum("cxy,xy->cxy", dx2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dx2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dx_E, dx_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dx_E_s.square())
            ) / E_s.pow(3)
            # e''' = (
            #   E'''(ΣE)³ - 3E''ΣE'(ΣE)² - 3E'(ΣE''(ΣE)²-2(ΣE')²ΣE)
            #   + E[6ΣE''ΣE'ΣE-ΣE'''(ΣE)²-6(ΣE')³]
            # )/(ΣE)⁴
            dx3_e = (
                torch.einsum("cxy,xy->cxy", dx3_E, E_s.pow(3))
                - 3 * torch.einsum("cxy,xy->cxy", dx2_E, dx_E_s * E_s.square())
                - 3
                * torch.einsum(
                    "cxy,xy->cxy",
                    dx_E,
                    dx2_E_s * E_s.square() - 2 * dx_E_s.square() * E_s,
                )
                + torch.einsum(
                    "cxy,xy->cxy",
                    E,
                    6 * dx2_E_s * dx_E_s * E_s
                    - dx3_E_s * E_s.square()
                    - 6 * dx_E_s.pow(3),
                )
            ) / E_s.pow(4)
            # ꟛ1 = e''' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dx3_e, gamma)
            # ꟛ2 = 3e'' γ'
            lambda2 = 3 * torch.einsum("cxy,cxyop->cxyop", dx2_e, dx_gamma)
            # ꟛ3 = 3e' γ''
            lambda3 = 3 * torch.einsum("cxy,cxyop->cxyop", dx_e, dx2_gamma)
            # ꟛ4 = e γ'''
            lambda4 = torch.einsum("cxy,cxyop->cxyop", e, dx3_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dx_gamma = -kx * torch.einsum(
                "o,cxyop->cxyop", self._cos_t, sin_xy
            )
            dy_gamma = -ky * torch.einsum(
                "o,cxyop->cxyop", self._sin_t, sin_xy
            )
            dx2_gamma = -(kx**2) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t.square(), gamma
            )
            dydx_gamma = -(kx * ky) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t * self._sin_t, gamma
            )
            dydx2_gamma = (kx**2 * ky) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t.square() * self._sin_t, sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dx_E = -2 * x / sx**2 * E
            dx_E_s = dx_E.sum(dim=0)
            dy_E = -2 * y / sy**2 * E
            dy_E_s = dy_E.sum(dim=0)
            dx2_E = (-2 / sx**2 + 4 * x**2 / sx**4) * E
            dx2_E_s = dx2_E.sum(dim=0)
            dydx_E = (-2 * y / sy**2) * dx_E
            dydx_E_s = dydx_E.sum(dim=0)
            dydx2_E = (-2 * y / sy**2) * dx2_E
            dydx2_E_s = dydx2_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # ∂_x e = (E_xΣE - EΣE_x)/(ΣE)²
            dx_e = (
                torch.einsum("cxy,xy->cxy", dx_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dx_E_s)
            ) / E_s.square()
            # ∂_y e = (E_yΣE - EΣE_y)/(ΣE)²
            dy_e = (
                torch.einsum("cxy,xy->cxy", dy_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dy_E_s)
            ) / E_s.square()
            # ∂_xx e = (E_xx(ΣE)² - EΣE_xxΣE - 2E_xΣE_xΣE + 2E(ΣE_x)²)/(ΣE)³
            dx2_e = (
                torch.einsum("cxy,xy->cxy", dx2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dx2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dx_E, dx_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dx_E_s.square())
            ) / E_s.pow(3)
            # ∂_xy e = (
            #   E_xy(ΣE)² + E_xΣE_yΣE - E_yΣE_xΣE -
            #   EΣE_xyΣE - 2E_xΣE_yΣE + 2 EΣE_xΣE_y
            # )/(ΣE)³
            dydx_e = (
                torch.einsum("cxy,xy->cxy", dydx_E, E_s.square())
                + torch.einsum("cxy,xy->cxy", dx_E, dy_E_s * E_s)
                - torch.einsum("cxy,xy->cxy", dy_E, dx_E_s * E_s)
                - torch.einsum("cxy,xy->cxy", E, dydx_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dx_E, dy_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dy_E_s * dx_E_s)
            ) / E_s.pow(3)
            # ∂_xxy e = (
            #   E_xxy(ΣE)³ + E(ΣE_xxΣE_yΣE - 4ΣE_y(ΣE_x)² - ΣE_xxy(ΣE)² +
            #   4ΣE_xyΣE_xΣE) + 2E_x(ΣE_xΣE_yΣE - ΣE_xy(ΣE)²) +
            #   E_yΣE(ΣE_xxΣE + (ΣE_x)²)
            # )/(ΣE)⁴
            dydx2_e = (
                torch.einsum("cxy,xy->cxy", dydx2_E, E_s.pow(3))
                + torch.einsum(
                    "cxy,xy->cxy",
                    E,
                    dx2_E_s * dy_E_s * E_s
                    - 4 * dy_E_s * dx_E_s.square()
                    - dydx2_E_s * E_s.square()
                    + 4 * dydx_E_s * dx_E_s * E_s,
                )
                + 2
                * torch.einsum(
                    "cxy,xy->cxy",
                    dx_E,
                    dx_E_s * dy_E_s * E_s - dydx_E_s * E_s.square(),
                )
                + torch.einsum(
                    "cxy,xy->cxy",
                    dy_E,
                    dx2_E_s * E_s.square() + dx_E_s.square() * E_s,
                )
            ) / E_s.pow(4)

            # ꟛ1 = ∂_xxy e γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dydx2_e, gamma)
            # ꟛ2 = ∂_xx e ∂_y γ
            lambda2 = torch.einsum("cxy,cxyop->cxyop", dx2_e, dy_gamma)
            # ꟛ3 = 2 ∂_xy e ∂_x γ
            lambda3 = 2 * torch.einsum("cxy,cxyop->cxyop", dydx_e, dx_gamma)
            # ꟛ4 = 2 ∂_x e ∂_xy γ
            lambda4 = 2 * torch.einsum("cxy,cxyop->cxyop", dx_e, dydx_gamma)
            # ꟛ5 = ∂_y e ∂_xx γ
            lambda5 = torch.einsum("cxy,cxyop->cxyop", dy_e, dx2_gamma)
            # ꟛ5 = e ∂_xxy γ
            lambda6 = torch.einsum("cxy,cxyop->cxyop", e, dydx2_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)
            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dy_gamma = -ky * torch.einsum(
                "o,cxyop->cxyop", self._sin_t, sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dy_E = -2 * y / sy**2 * E
            dy_E_s = dy_E.sum(dim=0)

            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dy_e = (
                torch.einsum("cxy,xy->cxy", dy_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dy_E_s)
            ) / E_s.square()
            # ꟛ1 = e' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dy_e, gamma)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e, dy_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dy_gamma = -ky * torch.einsum(
                "o,cxyop->cxyop", self._sin_t, sin_xy
            )
            dy2_gamma = -(ky**2) * torch.einsum(
                "o,cxyop->cxyop", self._sin_t.square(), gamma
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dy_E = -2 * y / sy**2 * E
            dy_E_s = dy_E.sum(dim=0)
            dy2_E = (-2 / sy**2 + 4 * y**2 / sy**4) * E
            dy2_E_s = dy2_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dy_e = (
                torch.einsum("cxy,xy->cxy", dy_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dy_E_s)
            ) / E_s.square()
            # e'' = (E''(ΣE)² - EΣE''ΣE - 2E'ΣE'ΣE + 2E(ΣE')²)/(ΣE)³
            dy2_e = (
                torch.einsum("cxy,xy->cxy", dy2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dy2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dy_E, dy_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dy_E_s.square())
            ) / E_s.pow(3)

            # ꟛ1 = e'' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dy2_e, gamma)
            # ꟛ2 = 2e' γ'
            lambda2 = 2 * torch.einsum("cxy,cxyop->cxyop", dy_e, dy_gamma)
            # ꟛ3 = e γ''
            lambda3 = torch.einsum("cxy,cxyop->cxyop", e, dy2_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dy_gamma = -ky * torch.einsum(
                "o,cxyop->cxyop", self._sin_t, sin_xy
            )
            dy2_gamma = -(ky**2) * torch.einsum(
                "o,cxyop->cxyop", self._sin_t.square(), gamma
            )
            dy3_gamma = (ky**3) * torch.einsum(
                "o,cxyop->cxyop", self._sin_t.pow(3), sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dy_E = -2 * y / sy**2 * E
            dy_E_s = dy_E.sum(dim=0)
            dy2_E = (-2 / sy**2 + 4 * y**2 / sy**4) * E
            dy2_E_s = dy2_E.sum(dim=0)
            dy3_E = (12 * y / sy**4 - 8 * y**3 / sy**6) * E
            dy3_E_s = dy3_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # e' = (E'ΣE - EΣE')/(ΣE)²
            dy_e = (
                torch.einsum("cxy,xy->cxy", dy_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dy_E_s)
            ) / E_s.square()
            # e'' = (E''(ΣE)² - EΣE''ΣE - 2E'ΣE'ΣE + 2E(ΣE')²)/(ΣE)³
            dy2_e = (
                torch.einsum("cxy,xy->cxy", dy2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dy2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dy_E, dy_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dy_E_s.square())
            ) / E_s.pow(3)
            # e''' = (
            #   E'''(ΣE)³ - 3E''ΣE'(ΣE)² - 3E'(ΣE''(ΣE)²-2(ΣE')²ΣE)
            #   + E[6ΣE''ΣE'ΣE-ΣE'''(ΣE)²-6(ΣE')³]
            # )/(ΣE)⁴
            dy3_e = (
                torch.einsum("cxy,xy->cxy", dy3_E, E_s.pow(3))
                - 3 * torch.einsum("cxy,xy->cxy", dy2_E, dy_E_s * E_s.square())
                - 3
                * torch.einsum(
                    "cxy,xy->cxy",
                    dy_E,
                    dy2_E_s * E_s.square() - 2 * dy_E_s.square() * E_s,
                )
                + torch.einsum(
                    "cxy,xy->cxy",
                    E,
                    6 * dy2_E_s * dy_E_s * E_s
                    - dy3_E_s * E_s.square()
                    - 6 * dy_E_s.pow(3),
                )
            ) / E_s.pow(4)

            # ꟛ1 = e''' γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dy3_e, gamma)
            # ꟛ2 = 3e'' γ'
            lambda2 = 3 * torch.einsum("cxy,cxyop->cxyop", dy2_e, dy_gamma)
            # ꟛ3 = 3e' γ''
            lambda3 = 3 * torch.einsum("cxy,cxyop->cxyop", dy_e, dy2_gamma)
            # ꟛ4 = e γ'''
            lambda4 = torch.einsum("cxy,cxyop->cxyop", e, dy3_gamma)
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

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dy_gamma = -ky * torch.einsum(
                "o,cxyop->cxyop", self._sin_t, sin_xy
            )
            dx_gamma = -kx * torch.einsum(
                "o,cxyop->cxyop", self._cos_t, sin_xy
            )
            dy2_gamma = -(ky**2) * torch.einsum(
                "o,cxyop->cxyop", self._sin_t.square(), gamma
            )
            dxdy_gamma = -(kx * ky) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t * self._sin_t, gamma
            )
            dxdy2_gamma = (ky**2 * kx) * torch.einsum(
                "o,cxyop->cxyop", self._cos_t * self._sin_t.square(), sin_xy
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            dy_E = -2 * y / sy**2 * E
            dy_E_s = dy_E.sum(dim=0)
            dx_E = -2 * x / sx**2 * E
            dx_E_s = dx_E.sum(dim=0)
            dy2_E = (-2 / sy**2 + 4 * y**2 / sy**4) * E
            dy2_E_s = dy2_E.sum(dim=0)
            dxdy_E = (-2 * x / sx**2) * dy_E
            dxdy_E_s = dxdy_E.sum(dim=0)
            dxdy2_E = (-2 * x / sx**2) * dy2_E
            dxdy2_E_s = dxdy2_E.sum(dim=0)
            # e = E/ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            # ∂_y e = (E_yΣE - EΣE_y)/(ΣE)²
            dy_e = (
                torch.einsum("cxy,xy->cxy", dy_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dy_E_s)
            ) / E_s.square()
            # ∂_x e = (E_xΣE - EΣE_x)/(ΣE)²
            dx_e = (
                torch.einsum("cxy,xy->cxy", dx_E, E_s)
                - torch.einsum("cxy,xy->cxy", E, dx_E_s)
            ) / E_s.square()
            # ∂_yy e = (E_yy(ΣE)² - EΣE_yyΣE - 2E_yΣE_yΣE + 2E(ΣE_y)²)/(ΣE)³
            dy2_e = (
                torch.einsum("cxy,xy->cxy", dy2_E, E_s.square())
                - torch.einsum("cxy,xy->cxy", E, dy2_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dy_E, dy_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dy_E_s.square())
            ) / E_s.pow(3)
            # ∂_yx e = (
            #   E_yx(ΣE)² + E_yΣE_xΣE - E_xΣE_yΣE -
            #   EΣE_yxΣE - 2E_yΣE_xΣE + 2 EΣE_yΣE_x
            # )/(ΣE)³
            dxdy_e = (
                torch.einsum("cxy,xy->cxy", dxdy_E, E_s.square())
                + torch.einsum("cxy,xy->cxy", dy_E, dx_E_s * E_s)
                - torch.einsum("cxy,xy->cxy", dx_E, dy_E_s * E_s)
                - torch.einsum("cxy,xy->cxy", E, dxdy_E_s * E_s)
                - 2 * torch.einsum("cxy,xy->cxy", dy_E, dx_E_s * E_s)
                + 2 * torch.einsum("cxy,xy->cxy", E, dy_E_s * dx_E_s)
            ) / E_s.pow(3)
            # ∂_yyx e = (
            #   E_yyx(ΣE)³ + E(ΣE_yyΣE_xΣE - 4ΣE_x(ΣE_y)² - ΣE_yyx(ΣE)² +
            #   4ΣE_yxΣE_yΣE) + 2E_y(ΣE_yΣE_xΣE - ΣE_yx(ΣE)²) +
            #   E_xΣE(ΣE_yyΣE + (ΣE_y)²)
            # )/(ΣE)⁴
            dxdy2_e = (
                torch.einsum("cxy,xy->cxy", dxdy2_E, E_s.pow(3))
                + torch.einsum(
                    "cxy,xy->cxy",
                    E,
                    dy2_E_s * dx_E_s * E_s
                    - 4 * dx_E_s * dy_E_s.square()
                    - dxdy2_E_s * E_s.square()
                    + 4 * dxdy_E_s * dy_E_s * E_s,
                )
                + 2
                * torch.einsum(
                    "cxy,xy->cxy",
                    dy_E,
                    dy_E_s * dx_E_s * E_s - dxdy_E_s * E_s.square(),
                )
                + torch.einsum(
                    "cxy,xy->cxy",
                    dx_E,
                    dy2_E_s * E_s.square() + dy_E_s.square() * E_s,
                )
            ) / E_s.pow(4)

            # ꟛ1 = ∂_yyx e γ
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dxdy2_e, gamma)
            # ꟛ2 = ∂_yy e ∂_x γ
            lambda2 = torch.einsum("cxy,cxyop->cxyop", dy2_e, dx_gamma)
            # ꟛ3 = 2 ∂_yx e ∂_y γ
            lambda3 = 2 * torch.einsum("cxy,cxyop->cxyop", dxdy_e, dy_gamma)
            # ꟛ4 = 2 ∂_y e ∂_yx γ
            lambda4 = 2 * torch.einsum("cxy,cxyop->cxyop", dy_e, dxdy_gamma)
            # ꟛ5 = ∂_x e ∂_yy γ
            lambda5 = torch.einsum("cxy,cxyop->cxyop", dx_e, dy2_gamma)
            # ꟛ6 = e ∂_yyx γ
            lambda6 = torch.einsum("cxy,cxyop->cxyop", e, dxdy2_gamma)
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
        field = torch.zeros_like(space_fields[0][0])
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
        field = torch.zeros_like(space_fields[0][0])
        tspecs = specs.from_tensor(t)
        for lvl, params in time_params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]
            space = space_fields[lvl]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_s = exp.sum(dim=0)
            dt_exp = -2 * (t - tc) / st**2 * exp

            field_at_lvl = (
                torch.einsum("t,txy->xy", dt_exp, space) * exp_s
                - torch.einsum("t,txy->xy", exp, space) * dt_exp.sum(dim=0)
            ) / exp_s**2

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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

        return OptimizableFunction(at_time)

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
