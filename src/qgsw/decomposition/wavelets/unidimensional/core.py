"""Wavelets implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qgsw.decomposition.wavelets.unidimensional.basis_functions import (
    CosineBasisFunctions,
)
from qgsw.decomposition.wavelets.unidimensional.param_generators import (
    dyadic_decomposition,
    linear_decomposition,
)
from qgsw.decomposition.wavelets.unidimensional.supports import (
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


class WaveletBasis1D:
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
        theta = torch.linspace(0, torch.pi, self.n_theta + 1, **self._specs)[
            :-1
        ]
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
        n = sum(
            s["numel"] * self._time[k]["numel"] for k, s in self._space.items()
        )

        return n * self.phase.numel() * self.n_theta

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
        for k in self._space:
            coefs[k] = torch.randn(
                (
                    self._time[k]["numel"],
                    self._space[k]["numel"],
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
        self, params: dict[str, Any], x: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        xc = params["centers"]
        kx = params["kx"]
        xc = torch.tensor(xc, **self._specs)

        x = x[None, :] - xc[:, None]
        return (x, kx)

    def _build_space(self, xx: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build space-related fields.

        Σe(x,y)ΣΣcγ(x,y)

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]

            x, kx = self._compute_space_params(params, xx)

            phase = self.phase[None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                kx,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, sx)
            e = NormalizedGaussianSupport(E)

            lambd = torch.einsum("cx,cxop->cxop", e.field, gamma.field)
            # ꟛ = e γ
            fields[lvl] = torch.einsum("tcop,cxop->txop", c, lambd).mean(
                dim=[-1, -2]
            )
        return fields

    def _build_space_dx(self, xx: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build x-derivatives of space fields.

        ΣΣΣc[e'(x,y)γ(x,y)+e'(x,y)γ'(x,y)]

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]

            x, kx = self._compute_space_params(params, xx)

            phase = self.phase[None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                kx,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, sx)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e' γ
            lambda1 = torch.einsum("cx,cxop->cxop", e.dx, gamma.field)
            # ꟛ2 = e γ'
            lambda2 = torch.einsum("cx,cxop->cxop", e.field, gamma.dx)
            # coefs * (e' γ + e γ')
            fields[lvl] = torch.einsum(
                "tcop,cxop->txop", coefs, lambda1 + lambda2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx2(self, xx: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build second order x-derivatives of space fields.

        ΣΣΣc[e''(x,y)γ(x,y) + 2e(x,y)'γ'(x,y) + e(x,y)γ''(x,y)]

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]

            x, kx = self._compute_space_params(params, xx)

            phase = self.phase[None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                kx,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, sx)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e'' γ
            lambda1 = torch.einsum("cx,cxop->cxop", e.dx2, gamma.field)
            # ꟛ2 = 2e' γ'
            lambda2 = 2 * torch.einsum("cx,cxop->cxop", e.dx, gamma.dx)
            # ꟛ3 = e γ''
            lambda3 = torch.einsum("cx,cxop->cxop", e.field, gamma.dx2)
            # coefs * (e'' γ +2e' γ'+ e γ'')
            fields[lvl] = torch.einsum(
                "tcop,cxop->txop", coefs, lambda1 + lambda2 + lambda3
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx3(self, xx: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build third order x-derivatives of space fields.

        ΣΣΣc[e'''γ + 3e''γ'+ 3e'γ'' +eγ''']

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]

            x, kx = self._compute_space_params(params, xx)

            phase = self.phase[None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                kx,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, sx)
            e = NormalizedGaussianSupport(E)

            # ꟛ1 = e''' γ
            lambda1 = torch.einsum("cx,cxop->cxop", e.dx3, gamma.field)
            # ꟛ2 = 3e'' γ'
            lambda2 = 3 * torch.einsum("cx,cxop->cxop", e.dx2, gamma.dx)
            # ꟛ3 = 3e' γ''
            lambda3 = 3 * torch.einsum("cx,cxop->cxop", e.dx, gamma.dx2)
            # ꟛ4 = e γ'''
            lambda4 = torch.einsum("cx,cxop->cxop", e.field, gamma.dx3)
            # coefs * (e''' γ + 3e' γ'' + 3 e' γ'' + e γ''')
            fields[lvl] = torch.einsum(
                "tcop,cxop->txop",
                coefs,
                lambda1 + lambda2 + lambda3 + lambda4,
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

            field_at_lvl = torch.einsum("t,tx->x", exp, space_fields[lvl])

            field += field_at_lvl
        return field / len(time_params)

    @staticmethod
    def _at_time_decompose(
        t: torch.Tensor,
        space_fields: dict[int, torch.Tensor],
        time_params: dict[int, dict[str, Any]],
    ) -> dict[int, torch.Tensor]:
        """Compute the decomposed field value at a given time.

        Args:
            t (torch.Tensor): Time to compute field at.
            space_fields (dict[int, torch.Tensor]): Space-only fields.
            time_params (dict[int, dict[str, Any]]): Time parameters.

        Returns:
            torch.Tensor: Resulting field.
        """
        fields = {}
        tspecs = specs.from_tensor(t)
        for lvl, params in time_params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)

            field_at_lvl = torch.einsum("t,tx->x", exp, space_fields[lvl])

            fields[lvl] = field_at_lvl
        return fields

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
            dt_exp = -2 * (t - tc) / st**2 * exp

            field_at_lvl = torch.einsum("t,tx->x", dt_exp, space)

            field += field_at_lvl
        return field / len(time_params)

    def localize(self, xx: torch.Tensor) -> WVFunc:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._at_time(t, space_fields, self._time)

        return at_time

    def localize_decompose(
        self, xx: torch.Tensor
    ) -> Callable[[torch.Tensor], dict[int, torch.Tensor]]:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._at_time_decompose(
                t, space_fields, self._time
            )

        return at_time

    def localize_dt(self, xx: torch.Tensor) -> WVFunc:
        """Localize wavelets time derivative.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._dt_at_time(t, space_fields, self._time)

        return at_time

    def localize_dx(self, xx: torch.Tensor) -> WVFunc:
        """Localize wavelets x-derivative.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._at_time(t, space_fields, self._time)

        return at_time

    def localize_dx2(self, xx: torch.Tensor) -> WVFunc:
        """Localize wavelets second order x-derivative.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx2(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._at_time(t, space_fields, self._time)

        return at_time

    def localize_dx3(self, xx: torch.Tensor) -> WVFunc:
        """Localize wavelets third order x-derivative.

        Args:
            xx (torch.Tensor): X locations.

        Returns:
            WVFunc: Function computing the wavelet field at a given time.
        """
        space_fields = self._build_space_dx3(xx=xx)

        def at_time(t: torch.Tensor) -> torch.Tensor:
            return WaveletBasis1D._at_time(t, space_fields, self._time)

        return at_time

    @classmethod
    def from_dyadic_decomposition(
        cls,
        order: int,
        x_ref: torch.Tensor,
        yy_ref: torch.Tensor,
        Lx_max: float,
        Lt_max: float,
    ) -> Self:
        """Generate space and time basis parameters to instantiate the object.

        Args:
            order (int): "Depth" of decomposition.
            x_ref (torch.Tensor): X locations to use as reference.
            yy_ref (torch.Tensor): Y locations to use as reference.
            Lx_max (float): Max horizontal scale.
            Lt_max (float): Max time scale.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: WaveletBasis1D.
        """
        space_params, time_params = dyadic_decomposition(
            order=order,
            x_ref=x_ref,
            yy_ref=yy_ref,
            Lx_max=Lx_max,
            Lt_max=Lt_max,
        )
        return cls(space_params, time_params)

    @classmethod
    def from_linear_decomposition(
        cls,
        order: int,
        x_ref: torch.Tensor,
        yy_ref: torch.Tensor,
        Lx_max: float,
        Lt_max: float,
    ) -> Self:
        """Generate space and time basis parameters to instantiate the object.

        Args:
            order (int): "Depth" of decomposition.
            x_ref (torch.Tensor): X locations to use as reference.
            yy_ref (torch.Tensor): Y locations to use as reference.
            Lx_max (float): Max horizontal scale.
            Lt_max (float): Max time scale.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: WaveletBasis1D.
        """
        space_params, time_params = linear_decomposition(
            order=order,
            x_ref=x_ref,
            yy_ref=yy_ref,
            Lx_max=Lx_max,
            Lt_max=Lt_max,
        )
        return cls(space_params, time_params)
