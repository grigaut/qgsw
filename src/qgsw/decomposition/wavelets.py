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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            sx = params["sigma_x"]
            sy = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            gamma_xy = torch.cos(
                (kx_cos + ky_sin)[..., None]
                + self.phase[None, None, None, None, :]
            )

            E = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            E_s = E.sum(dim=0)
            # e = E / ΣE  # noqa: ERA001
            e = torch.einsum("cxy,xy->cxy", E, 1 / E_s)
            lambd = torch.einsum("cxy,cxyop->cxyop", e, gamma_xy)
            # ꟛ = e γ
            fields[lvl] = torch.einsum("tcop,cxyop->txyop", c, lambd).mean(
                dim=[-1, -2]
            )
        return fields

    def _build_space_dx(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> torch.Tensor:
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
            sx = params["sigma_x"]
            sy = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)

            phase = self.phase[None, None, None, None, :]

            gamma_xy = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dx_gamma_xy = -kx * torch.einsum(
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
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dx_e, gamma_xy)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e, dx_gamma_xy)
            # coefs * (e' γ + e γ')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> torch.Tensor:
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
            sx = params["sigma_x"]
            sy = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            kx_cos = kx * torch.einsum("cxy,o->cxyo", x, self._cos_t)
            ky_sin = ky * torch.einsum("cxy,o->cxyo", y, self._sin_t)
            phase = self.phase[None, None, None, None, :]

            gamma_xy = torch.cos((kx_cos + ky_sin)[..., None] + phase)
            sin_xy = torch.sin((kx_cos + ky_sin)[..., None] + phase)
            dy_gamma_xy = -ky * torch.einsum(
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
            lambda1 = torch.einsum("cxy,cxyop->cxyop", dy_e, gamma_xy)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cxy,cxyop->cxyop", e, dy_gamma_xy)
            # coefs * (e' γ + e γ')
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, lambda1 + lambda2
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
            st = params["sigma_t"]

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
            st = params["sigma_t"]
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
