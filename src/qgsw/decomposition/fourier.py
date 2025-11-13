"""Cos / sine decomposition."""

from __future__ import annotations

import torch

from qgsw.logging import getLogger
from qgsw.specs import defaults

logger = getLogger(__name__)


class FourierBasis:
    """Cos / Sine basis."""

    _coefs: torch.Tensor = None
    _c = 0.5

    @property
    def order(self) -> int:
        """Decomposition order."""
        return self._order

    def __init__(
        self,
        xx: torch.Tensor,
        yy: torch.Tensor,
        *,
        order: int = 4,
        Lx_max: float | None = None,  # noqa: N803
        Ly_max: float | None = None,  # noqa: N803
    ) -> None:
        """Instantiate the Basis.

        Args:
            xx (torch.Tensor): Xs.
            yy (torch.Tensor): Ys.
            order (int, optional): Decomposition order. Defaults to 4.
            Lx_max (float | None, optional): Largest dimension along X,
                total width if set to None. Defaults to None.
            Ly_max (float | None, optional): Largest dimension along Y,
                total width if set to None. Defaults to None.
        """
        self._order = order
        self._x = xx
        self._lx = xx[-1, 0] - xx[0, 0]
        self._y = yy
        self._ly = yy[0, -1] - yy[0, 0]
        self._order = order
        self._Lx = Lx_max if Lx_max is not None else self._lx
        self._Ly = Ly_max if Ly_max is not None else self._ly
        self._generate_spatial_basis(order)
        self._generate_time_basis()

    def numel(self) -> int:
        """Total number of elements."""
        return self._K.numel()

    def __repr__(self) -> str:
        """Strin representation."""
        return f"CosSineBasis(order={self.order})"

    def _generate_spatial_basis(
        self,
        order: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Generate space basis.

        Args:
            order (int): Order.
            dtype (torch.dtype | None, optional): Dtype. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        specs = defaults.get(dtype=dtype, device=device)
        ks = 2 * torch.pi * (1 + torch.arange(order, **specs)) / (self._Lx)
        ls = 2 * torch.pi * (1 + torch.arange(order, **specs)) / (self._Ly)

        self._K, self._L = torch.meshgrid(ks, ls, indexing="ij")

        self._exp_space = torch.exp(
            1j
            * (
                self._K[..., None, None] * self._x[None, None, ...]
                + self._L[..., None, None] * self._y[None, None, ...]
            )
        )

    def _generate_time_basis(
        self,
    ) -> None:
        """Generate ω."""
        self._omega = (
            self._c
            * (self._K.square() + self._L.square()).sqrt()[..., None, None]
        )

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
        exp_time = torch.exp(-1j * self._omega * t[None, None, None, :])
        return (
            (self._coefs[..., None, None] * self._exp_space * exp_time)
            .sum(dim=[0, 1])
            .real
        )

    def set_coefs(self, coefs: torch.Tensor) -> None:
        """Set coefficients values.

        To ensure proper coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (torch.Tensor): Coefficients.
        """
        self._coefs = coefs

    def generate_random_coefs(self) -> torch.Tensor:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            torch.Tensor: Coefficients.
        """
        return torch.rand_like(self._K)
