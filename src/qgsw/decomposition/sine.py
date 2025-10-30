"""Sine decomposition."""

from __future__ import annotations

import itertools

import torch

from qgsw.logging import getLogger
from qgsw.specs import defaults

logger = getLogger(__name__)


class STSineBasis:
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
        self.set_coefs(self._coefs)

    @property
    def order(self) -> int:
        """Decomposition order."""
        return self._order

    def numel(self) -> int:
        """Total number of elements."""
        return sum((2**i) ** 3 for i in range(self._order))

    def __repr__(self) -> str:
        """Strin representation."""
        return f"STSineBasis(order={self.order}, normalize={self.normalize})"

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

            sigma_x = lx * 0.5  # For the gaussian enveloppe
            sigma_y = ly * 0.5  # For the gaussian enveloppe

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

            sigma_t = lt * 0.5  # For the gaussian enveloppe

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
                ├── 0: (1, 1)-shaped
                ├── 1: (2, 4)-shaped
                ├── 2: (4, 16)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2)-shaped
            level (int): level to build.

        Returns:
            torch.Tensor: Space field at the given level.
        """
        field = torch.zeros_like(self._x)
        norm = torch.zeros_like(self._x)

        centers = self.space_basis[level]["centers"]
        kx_p = self.space_basis[level]["kx"]
        ky_p = self.space_basis[level]["ky"]
        sx = self.space_basis[level]["sigma_x"]
        sy = self.space_basis[level]["sigma_y"]

        xx = self._x
        yy = self._y

        for i, xy in enumerate(centers):
            xc, yc = xy
            x = xx - xc
            y = yy - yc
            coef = coefs[i]
            exp = torch.exp(-((x**2) / (sx) ** 2 + (y**2) / (sy) ** 2))
            field += exp * (torch.sin(x * kx_p) + torch.sin(y * ky_p)) * coef
            norm += exp
        if self.normalize:
            field /= norm
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
            field += (exp[:, None, None] * self._fields[lvl]).sum(dim=0)
            if self.normalize:
                field /= exp.sum(dim=0)
        if self.normalize:
            field /= self.order + 1
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
                (self.time_basis[o]["numel"], self.space_basis[o]["numel"]),
                **defaults.get(dtype=dtype, device=device),
            )
        return coefs
