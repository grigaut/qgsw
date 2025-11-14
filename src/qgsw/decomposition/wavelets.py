"""Sine decomposition."""

from __future__ import annotations

import itertools

import torch

import qgsw
import qgsw.specs
from qgsw.logging import getLogger
from qgsw.specs import defaults

logger = getLogger(__name__)


class WaveletBasis:
    """Space-Time sine basis.

    Space is subdivided in patches of size Lx_max / 2**p x Ly_max / 2**p
    for p in [0, order-1]. Each patch is associated with a temporal Gaussian
    enveloppe centered on the middle of time patches of size Lt_max / 2**p.

    The patches are not stricly confined to their domain, but have a gaussian
    enveloppe that decays away from the center of the patch.

    Hence, at order 0, there is one patch covering the whole domain, associated
    with a gaussian centered in the middle of the time domain. At order 1,
    there are 4 patches associated with a gaussian centered a 1/4 of the time
    domain, and 4 other patches associated with a gaussian centered at 3/4 of
    the time domain, etc.

    Hence, at order 'p', there are (2**p)x(2**p) spatial patches, each
    associated with 2**p temporal gaussian enveloppes, for a total of (2**p)**3
    basis elements at order p.
    """

    _normalize = True
    _coefs: torch.Tensor = None
    _n_theta = 10
    _dx_fields = None
    _dy_fields = None
    _sigma_ratio = torch.sqrt(torch.log(torch.tensor(2.0))).item()

    def __init__(
        self,
        xx: torch.Tensor,
        yy: torch.Tensor,
        tt: torch.Tensor,
        *,
        order: int = 4,
        Lx_max: float | None = None,  # noqa: N803
        Ly_max: float | None = None,  # noqa: N803
        Lt_max: float | None = None,  # noqa: N803
    ) -> None:
        """Instantiate the Basis.

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
        """
        self._order = order
        self._x = xx
        self._lx = xx[-1, 0] - xx[0, 0]
        self._y = yy
        self._ly = yy[0, -1] - yy[0, 0]
        self._t = tt
        self._lt = tt[-1] - tt[0]
        self._order = order
        self._Lx = Lx_max if Lx_max is not None else self._lx
        self._Ly = Ly_max if Ly_max is not None else self._ly
        self._Lt = Lt_max if Lt_max is not None else self._lt
        self._generate_spatial_basis(order)
        self._generate_time_basis(order)

    @property
    def normalize(self) -> bool:
        """Whether to normalize the output."""
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        if normalize == self._normalize:
            return
        self._normalize = normalize
        if self._coefs is not None:
            self.set_coefs(self._coefs)

    @property
    def order(self) -> int:
        """Decomposition order."""
        return self._order

    @property
    def n_theta(self) -> int:
        """Number of directions."""
        return self._n_theta

    @n_theta.setter
    def n_theta(self, n_theta: int) -> None:
        self._n_theta = n_theta

    def numel(self) -> int:
        """Total number of elements."""
        return sum((2**i) ** 3 for i in range(self._order)) * 2 * self.n_theta

    def __repr__(self) -> str:
        """Strin representation."""
        return f"WaveletBasis(order={self.order}, normalize={self.normalize})"

    def _generate_spatial_basis(self, order: int) -> None:
        """Generate spatial basis.

        Args:
            order (int): Decomposition order.
        """
        basis = {}
        for p in range(order):
            Lx_p = self._Lx / 2**p  # noqa: N806
            Ly_p = self._Ly / 2**p  # noqa: N806
            kx_p = 2 * torch.pi / Lx_p
            ky_p = 2 * torch.pi / Ly_p

            lx = (self._lx) / 2**p
            xs = [self._x[0, 0] + (2 * k + 1) / 2 * lx for k in range(2**p)]
            ly = (self._ly) / 2**p
            ys = [self._y[0, 0] + (2 * k + 1) / 2 * ly for k in range(2**p)]

            centers = [
                (x.cpu().item(), y.cpu().item())
                for x, y in itertools.product(xs, ys)
            ]

            sigma_x = lx / self._sigma_ratio  # For the gaussian enveloppe
            sigma_y = ly / self._sigma_ratio  # For the gaussian enveloppe

            basis[p] = {
                "centers": centers,
                "kx": kx_p,
                "ky": ky_p,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "numel": len(centers),
            }

        self.space_basis = basis

    def _generate_time_basis(self, order: int) -> None:
        """Generate time basis.

        Args:
            order (int): Decomposition order.
        """
        basis = {}
        for p in range(order):
            lt = (self._lt) / 2**p
            ts = [self._t[0] + (2 * k + 1) / 2 * lt for k in range(2**p)]

            centers = [t.cpu().item() for t in ts]

            sigma_t = lt / self._sigma_ratio  # For the gaussian enveloppe

            basis[p] = {
                "centers": centers,
                "sigma_t": sigma_t,
                "numel": len(centers),
            }
        self.time_basis = basis

    def _build_field(
        self,
        coefs: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Builds space field given coefficients.

        Args:
            coefs (torch.Tensor): Coefficients.
                ├── 0: (1, 1, n_theta, n_theta)-shaped
                ├── 1: (2, 4, n_theta, n_theta)-shaped
                ├── 2: (4, 16, n_theta, n_theta)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2, n_theta, n_theta)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2, n_theta, n_theta)-shaped
            level (int): level to build.

        Returns:
            torch.Tensor: Space field at the given level.
        """
        field = torch.zeros_like(self._x)

        centers = self.space_basis[level]["centers"]
        kx_p = self.space_basis[level]["kx"]
        ky_p = self.space_basis[level]["ky"]
        sx = self.space_basis[level]["sigma_x"]
        sy = self.space_basis[level]["sigma_y"]

        xx = self._x
        yy = self._y

        tspecs = qgsw.specs.from_tensor(xx)
        theta = torch.linspace(0, torch.pi, self.n_theta, **tspecs)
        phase = torch.tensor([0, torch.pi / 2], **tspecs)

        xc = torch.tensor([c[0] for c in centers], **tspecs)
        yc = torch.tensor([c[1] for c in centers], **tspecs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]

        exp = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))

        cos_t = torch.cos(theta)
        ct = cos_t[None, None, None, :, None]
        sine_t = torch.sin(theta)
        st = sine_t[None, None, None, :, None]

        p = phase[None, None, None, None, :]

        kx_cos = kx_p * x[..., None, None] * ct
        ky_sin = ky_p * y[..., None, None] * st

        cos_xy = torch.cos(kx_cos + ky_sin + p)

        coef_cos = cos_xy * coefs[:, None, None, :, :]

        mean_coef_cos = (coef_cos).mean(dim=[-1, -2])

        if self.normalize:
            field = (exp * mean_coef_cos).sum(dim=0) / (exp.sum(dim=0))
        else:
            field = (exp * mean_coef_cos).sum(dim=0)
        return field

    def _build_dx_field(
        self,
        coefs: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Build space field x-derivative given coefficients.

        Args:
            coefs (torch.Tensor): Coefficients.
                ├── 0: (1, 1, n_theta, n_theta)-shaped
                ├── 1: (2, 4, n_theta, n_theta)-shaped
                ├── 2: (4, 16, n_theta, n_theta)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2, n_theta, n_theta)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2, n_theta, n_theta)-shaped
            level (int): level to build.

        Returns:
            torch.Tensor: Space field x derivative at the given level.
        """
        field = torch.zeros_like(self._x)

        centers = self.space_basis[level]["centers"]
        kx_p = self.space_basis[level]["kx"]
        ky_p = self.space_basis[level]["ky"]
        sx = self.space_basis[level]["sigma_x"]
        sy = self.space_basis[level]["sigma_y"]

        xx = self._x
        yy = self._y

        tspecs = qgsw.specs.from_tensor(xx)
        theta = torch.linspace(0, torch.pi, self.n_theta, **tspecs)
        phase = torch.tensor([0, torch.pi / 2], **tspecs)

        xc = torch.tensor([c[0] for c in centers], **tspecs)
        yc = torch.tensor([c[1] for c in centers], **tspecs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]

        exp = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
        dx_exp = -2 * x / sx**2 * exp

        cos_t = torch.cos(theta)
        ct = cos_t[None, None, None, :, None]
        sine_t = torch.sin(theta)
        st = sine_t[None, None, None, :, None]

        p = phase[None, None, None, None, :]

        kx_cos = kx_p * x[..., None, None] * ct
        ky_sin = ky_p * y[..., None, None] * st

        cos_xy = torch.cos(kx_cos + ky_sin + p)
        dx_cos_xy = -kx_p * ct * torch.sin(kx_cos + ky_sin + p)

        coef_cos = cos_xy * coefs[:, None, None, :, :]
        dx_coef_cos = dx_cos_xy * coefs[:, None, None, :, :]

        mean_coef_cos = (coef_cos).mean(dim=[-1, -2])
        dx_mean_coef_cos = (dx_coef_cos).mean(dim=[-1, -2])

        if self.normalize:
            field = (
                (dx_exp * mean_coef_cos + exp * dx_mean_coef_cos).sum(dim=0)
                * exp.sum(dim=0)
                - (exp * mean_coef_cos).sum(dim=0) * dx_exp.sum(dim=0)
            ) / (exp.sum(dim=0) ** 2)
        else:
            field = (dx_exp * mean_coef_cos + exp * dx_mean_coef_cos).sum(
                dim=0
            )
        return field

    def _build_dy_field(
        self,
        coefs: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Build space field y-derivative given coefficients.

        Args:
            coefs (torch.Tensor): Coefficients.
                ├── 0: (1, 1, n_theta, n_theta)-shaped
                ├── 1: (2, 4, n_theta, n_theta)-shaped
                ├── 2: (4, 16, n_theta, n_theta)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2, n_theta, n_theta)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2, n_theta, n_theta)-shaped
            level (int): level to build.

        Returns:
            torch.Tensor: Space field y derivative at the given level.
        """
        field = torch.zeros_like(self._x)

        centers = self.space_basis[level]["centers"]
        kx_p = self.space_basis[level]["kx"]
        ky_p = self.space_basis[level]["ky"]
        sx = self.space_basis[level]["sigma_x"]
        sy = self.space_basis[level]["sigma_y"]

        xx = self._x
        yy = self._y

        tspecs = qgsw.specs.from_tensor(xx)
        theta = torch.linspace(0, torch.pi, self.n_theta, **tspecs)
        phase = torch.tensor([0, torch.pi / 2], **tspecs)

        xc = torch.tensor([c[0] for c in centers], **tspecs)
        yc = torch.tensor([c[1] for c in centers], **tspecs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]

        exp = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
        dy_exp = -2 * y / sy**2 * exp

        cos_t = torch.cos(theta)
        ct = cos_t[None, None, None, :, None]
        sine_t = torch.sin(theta)
        st = sine_t[None, None, None, :, None]

        p = phase[None, None, None, None, :]

        kx_cos = kx_p * x[..., None, None] * ct
        ky_sin = ky_p * y[..., None, None] * st

        cos_xy = torch.cos(kx_cos + ky_sin + p)
        dy_cos_xy = -ky_p * st * torch.sin(kx_cos + ky_sin + p)

        coef_cos = cos_xy * coefs[:, None, None, :, :]
        dy_coef_cos = dy_cos_xy * coefs[:, None, None, :, :]

        mean_coef_cos = (coef_cos).mean(dim=[-1, -2])
        dy_mean_coef_cos = (dy_coef_cos).mean(dim=[-1, -2])

        if self.normalize:
            field = (
                (dy_exp * mean_coef_cos + exp * dy_mean_coef_cos).sum(dim=0)
                * exp.sum(dim=0)
                - (exp * mean_coef_cos).sum(dim=0) * dy_exp.sum(dim=0)
            ) / (exp.sum(dim=0) ** 2)
        else:
            field = (dy_exp * mean_coef_cos + exp * dy_mean_coef_cos).sum(
                dim=0
            )
        return field

    def at_time(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Build the field at a given time.

        Args:
            t (torch.Tensor): Time.
                └── (1,)-shaped

        Returns:
            torch.Tensor: Build field.
        """
        field = torch.zeros_like(self._x)
        for lvl, base_elements in self.time_basis.items():
            centers = base_elements["centers"]
            st = base_elements["sigma_t"]

            exp = torch.cat(
                [torch.exp(-((t - tc) ** 2) / (st) ** 2) for tc in centers],
                dim=0,
            )
            field_at_lvl = (exp[:, None, None] * self._fields[lvl]).sum(dim=0)
            if self.normalize:
                field_at_lvl /= exp.sum(dim=0)
            field += field_at_lvl
        if self.normalize:
            field = field / (self.order)
        return field

    def set_coefs(
        self,
        coefs: dict[int, torch.Tensor],
    ) -> None:
        """Set coefficients values.

        To ensure proper coefficients shapes, best is to use
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
        self._fields: dict[int, torch.Tensor] = {}
        for lvl in coefs:
            self._fields[lvl] = torch.stack(
                [
                    self._build_field(coefs[lvl][i], level=lvl)
                    for i in range(coefs[lvl].shape[0])
                ],
                dim=0,
            )
        self._dx_fields = None
        self._dy_fields = None

    def generate_random_coefs(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> dict[int, torch.Tensor]:
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
                    self.time_basis[o]["numel"],
                    self.space_basis[o]["numel"],
                    self.n_theta,
                    2,
                ),
                **defaults.get(dtype=dtype, device=device),
            )
        return coefs

    def dt_at_time(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Build the time derivative of the field at a given time.

        Args:
            t (torch.Tensor): Time.
                └── (1,)-shaped

        Returns:
            torch.Tensor: Time derivative field.
        """
        field = torch.zeros_like(self._x)
        for lvl, base_elements in self.time_basis.items():
            centers = base_elements["centers"]
            st = base_elements["sigma_t"]

            exp = torch.cat(
                [torch.exp(-((t - tc) ** 2) / (st) ** 2) for tc in centers],
                dim=0,
            )
            dt_exp = torch.cat(
                [
                    -2
                    * ((t - tc) / st**2)
                    * torch.exp(-((t - tc) ** 2) / (st) ** 2)
                    for tc in centers
                ],
                dim=0,
            )
            field_at_lvl = (dt_exp[:, None, None] * self._fields[lvl]).sum(
                dim=0
            ) * exp.sum(dim=0) - (exp[:, None, None] * self._fields[lvl]).sum(
                dim=0
            ) * dt_exp.sum(dim=0)
            if self.normalize:
                field_at_lvl /= (exp.sum(dim=0)) ** 2
            field += field_at_lvl
        if self.normalize:
            field = field / (self.order)
        return field

    def dx_at_time(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Build the field at a given time.

        Args:
            t (torch.Tensor): Time.
                └── (1,)-shaped

        Returns:
            torch.Tensor: Build field.
        """
        if self._dx_fields is None:
            self._dx_fields: dict[int, torch.Tensor] = {}
            for lvl in self._coefs:
                self._dx_fields[lvl] = torch.stack(
                    [
                        self._build_dx_field(self._coefs[lvl][i], level=lvl)
                        for i in range(self._coefs[lvl].shape[0])
                    ],
                    dim=0,
                )
        field = torch.zeros_like(self._x)
        for lvl, base_elements in self.time_basis.items():
            centers = base_elements["centers"]
            st = base_elements["sigma_t"]

            exp = torch.cat(
                [torch.exp(-((t - tc) ** 2) / (st) ** 2) for tc in centers],
                dim=0,
            )
            field_at_lvl = (exp[:, None, None] * self._dx_fields[lvl]).sum(
                dim=0
            )
            if self.normalize:
                field_at_lvl /= exp.sum(dim=0)
            field += field_at_lvl
        if self.normalize:
            field = field / (self.order)
        return field

    def dy_at_time(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Build the field at a given time.

        Args:
            t (torch.Tensor): Time.
                └── (1,)-shaped

        Returns:
            torch.Tensor: Build field.
        """
        if self._dy_fields is None:
            self._dy_fields: dict[int, torch.Tensor] = {}
            for lvl in self._coefs:
                self._dy_fields[lvl] = torch.stack(
                    [
                        self._build_dy_field(self._coefs[lvl][i], level=lvl)
                        for i in range(self._coefs[lvl].shape[0])
                    ],
                    dim=0,
                )
        field = torch.zeros_like(self._x)
        for lvl, base_elements in self.time_basis.items():
            centers = base_elements["centers"]
            st = base_elements["sigma_t"]

            exp = torch.cat(
                [torch.exp(-((t - tc) ** 2) / (st) ** 2) for tc in centers],
                dim=0,
            )
            field_at_lvl = (exp[:, None, None] * self._dy_fields[lvl]).sum(
                dim=0
            )
            if self.normalize:
                field_at_lvl /= exp.sum(dim=0)
            field += field_at_lvl
        if self.normalize:
            field = field / (self.order)
        return field
