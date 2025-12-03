"""1D support for wavelets."""

from functools import cached_property

import torch


class GaussianSupport:
    """Space gaussian kernel."""

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

    def __init__(self, x: torch.Tensor, sx: float) -> None:
        """Instantiate the class.

        Args:
            x (torch.Tensor): X locations.
            sx (float): Sigma x.
        """
        self.x = x
        self.sx = sx

    def _compute(self) -> torch.Tensor:
        return torch.exp(-(self.x**2) / self.sx**2)

    def _compute_dx(self) -> torch.Tensor:
        return -2 * self.x / self.sx**2 * self.field

    def _compute_dx2(self) -> torch.Tensor:
        return (-2 / self.sx**2 + 4 * self.x**2 / self.sx**4) * self.field

    def _compute_dx3(self) -> torch.Tensor:
        return (
            12 * self.x / self.sx**4 - 8 * self.x**3 / self.sx**6
        ) * self.field


class SumGaussianSupports:
    """Center-sumed gaussian supports."""

    @cached_property
    def field(self) -> torch.Tensor:
        """Field."""
        return self.gs.field.sum(dim=0)

    @cached_property
    def dx(self) -> torch.Tensor:
        """X-derivative."""
        return self.gs.dx.sum(dim=0)

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self.gs.dx2.sum(dim=0)

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self.gs.dx3.sum(dim=0)

    def __init__(self, gaussian_support: GaussianSupport) -> None:
        """Instantiate the class.

        Args:
            gaussian_support (GaussianSupport): Gaussian support.
        """
        self.gs = gaussian_support


class NormalizedGaussianSupport:
    """Normalized space gaussian supports."""

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

    def __init__(self, gaussian_support: GaussianSupport) -> None:
        """Instantiate the class.

        Args:
            gaussian_support (GaussianSupport): Gaussian support.
        """
        self.gs = gaussian_support
        self.gs_s = SumGaussianSupports(gaussian_support)

    def _compute(self) -> torch.Tensor:
        return torch.einsum("cx,x->cx", self.gs.field, 1 / self.gs_s.field)

    def _compute_dx(self) -> torch.Tensor:
        t = torch.einsum(
            "cx,x->cx", self.gs.dx, self.gs_s.field
        ) - torch.einsum("cx,x->cx", self.gs.field, self.gs_s.dx)
        return torch.einsum("cx,x->cx", t, 1 / self.gs_s.field.pow(2))

    def _compute_dx2(self) -> torch.Tensor:
        t = (
            torch.einsum("cx,x->cx", self.gs.dx2, self.gs_s.field.pow(2))
            - torch.einsum(
                "cx,x->cx", self.gs.field, self.gs_s.dx2 * self.gs_s.field
            )
            - 2
            * torch.einsum(
                "cx,x->cx", self.gs.dx, self.gs_s.dx * self.gs_s.field
            )
            + 2 * torch.einsum("cx,x->cx", self.gs.field, self.gs_s.dx.pow(2))
        )
        return torch.einsum("cx,x->cx", t, 1 / self.gs_s.field.pow(3))

    def _compute_dx3(self) -> torch.Tensor:
        t = (
            torch.einsum("cx,x->cx", self.gs.dx3, self.gs_s.field.pow(3))
            - 3
            * torch.einsum(
                "cx,x->cx",
                self.gs.dx2,
                self.gs_s.dx * self.gs_s.field.pow(2),
            )
            + 3
            * torch.einsum(
                "cx,x->cx",
                self.gs.dx,
                -self.gs_s.dx2 * self.gs_s.field.pow(2)
                + 2 * self.gs_s.dx.pow(2) * self.gs_s.field,
            )
            + torch.einsum(
                "cx,x->cx",
                self.gs.field,
                -self.gs_s.dx3 * self.gs_s.field.pow(2)
                + 6 * self.gs_s.dx2 * self.gs_s.dx * self.gs_s.field
                - 6 * self.gs_s.dx.pow(3),
            )
        )
        return torch.einsum("cx,x->cx", t, 1 / self.gs_s.field.pow(4))
