"""Wavelet basis functions."""

from functools import cached_property

import torch


class CosineBasisFunctions:
    """Cosine basis functions."""

    @cached_property
    def field(self) -> torch.Tensor:
        """Field."""
        return self._compute()

    @cached_property
    def dx(self) -> torch.Tensor:
        """X-derivative."""
        return self._compute_dx()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    def __init__(
        self,
        x: torch.Tensor,
        kx: float,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Instantiate the class.

        Args:
            x (torch.Tensor): X locations.
            kx (float): X wave number.
            cos_t (torch.Tensor): Cosine of orientations.
            sin_t (torch.Tensor): Sine of orientations.
            phase (torch.Tensor): Phases.
        """
        kx_cos = kx * torch.einsum("cx,o->cxo", x, cos_t)
        self.cos_t = cos_t
        self.sin_t = sin_t
        self.cos = torch.cos((kx_cos)[..., None] + phase)
        self.sin = torch.sin((kx_cos)[..., None] + phase)
        self.kx = kx

    def _compute(self) -> torch.Tensor:
        return self.cos

    def _compute_dx(self) -> torch.Tensor:
        return -self.kx * torch.einsum("o,cxop->cxop", self.cos_t, self.sin)

    def _compute_dx2(self) -> torch.Tensor:
        return -(self.kx**2) * torch.einsum(
            "o,cxop->cxop", self.cos_t.pow(2), self.cos
        )

    def _compute_dx3(self) -> torch.Tensor:
        return (self.kx**3) * torch.einsum(
            "o,cxop->cxop", self.cos_t.pow(3), self.sin
        )
