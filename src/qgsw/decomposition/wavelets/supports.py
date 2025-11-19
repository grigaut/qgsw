"""Gaussian support."""

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
    def dy(self) -> torch.Tensor:
        """Y-derivative."""
        return self._compute_dy()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dydx(self) -> torch.Tensor:
        """X-Y-derivative."""
        return self._compute_dydx()

    @cached_property
    def dy2(self) -> torch.Tensor:
        """Second Y-derivative."""
        return self._compute_dy2()

    @cached_property
    def dxdy(self) -> torch.Tensor:
        """Y-X-derivative."""
        return self._compute_dxdy()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    @cached_property
    def dy3(self) -> torch.Tensor:
        """Third Y-derivative."""
        return self._compute_dy3()

    @cached_property
    def dydx2(self) -> torch.Tensor:
        """X-X-Y-derivative."""
        return self._compute_dydx2()

    @cached_property
    def dxdy2(self) -> torch.Tensor:
        """Y-Y-X-derivative."""
        return self._compute_dxdy2()

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, sx: float, sy: float
    ) -> None:
        """Instantiate the class.

        Args:
            x (torch.Tensor): X locations.
            y (torch.Tensor): Y locations.
            sx (float): Sigma x.
            sy (float): Sigma y.
        """
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy

    def _compute(self) -> torch.Tensor:
        return torch.exp(-(self.x**2) / self.sx**2 - self.y**2 / self.sy**2)

    def _compute_dx(self) -> torch.Tensor:
        return -2 * self.x / self.sx**2 * self.field

    def _compute_dy(self) -> torch.Tensor:
        return -2 * self.y / self.sy**2 * self.field

    def _compute_dx2(self) -> torch.Tensor:
        return (-2 / self.sx**2 + 4 * self.x**2 / self.sx**4) * self.field

    def _compute_dy2(self) -> torch.Tensor:
        return (-2 / self.sy**2 + 4 * self.y**2 / self.sy**4) * self.field

    def _compute_dydx(self) -> torch.Tensor:
        return (4 * self.x * self.y / self.sx**2 / self.sy**2) * self.field

    def _compute_dxdy(self) -> torch.Tensor:
        return self._compute_dydx()

    def _compute_dx3(self) -> torch.Tensor:
        return (
            12 * self.x / self.sx**4 - 8 * self.x**3 / self.sx**6
        ) * self.field

    def _compute_dydx2(self) -> torch.Tensor:
        return (-2 * self.y / self.sy**2) * self.dx2

    def _compute_dy3(self) -> torch.Tensor:
        return (
            12 * self.x / self.sx**4 - 8 * self.x**3 / self.sx**6
        ) * self.field

    def _compute_dxdy2(self) -> torch.Tensor:
        return (-2 * self.x / self.sx**2) * self.dy2


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
    def dy(self) -> torch.Tensor:
        """Y-derivative."""
        return self.gs.dy.sum(dim=0)

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self.gs.dx2.sum(dim=0)

    @cached_property
    def dydx(self) -> torch.Tensor:
        """X-Y-derivative."""
        return self.gs.dydx.sum(dim=0)

    @cached_property
    def dy2(self) -> torch.Tensor:
        """Second Y-derivative."""
        return self.gs.dy2.sum(dim=0)

    @cached_property
    def dxdy(self) -> torch.Tensor:
        """Y-X-derivative."""
        return self.gs.dxdy.sum(dim=0)

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self.gs.dx3.sum(dim=0)

    @cached_property
    def dy3(self) -> torch.Tensor:
        """Third Y-derivative."""
        return self.gs.dy3.sum(dim=0)

    @cached_property
    def dydx2(self) -> torch.Tensor:
        """X-X-Y-derivative."""
        return self.gs.dydx2.sum(dim=0)

    @cached_property
    def dxdy2(self) -> torch.Tensor:
        """Y-Y-X-derivative."""
        return self.gs.dxdy2.sum(dim=0)

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
    def dy(self) -> torch.Tensor:
        """Y-derivative."""
        return self._compute_dy()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dydx(self) -> torch.Tensor:
        """X-Y-derivative."""
        return self._compute_dydx()

    @cached_property
    def dy2(self) -> torch.Tensor:
        """Second Y-derivative."""
        return self._compute_dy2()

    @cached_property
    def dxdy(self) -> torch.Tensor:
        """Y-X-derivative."""
        return self._compute_dxdy()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    @cached_property
    def dy3(self) -> torch.Tensor:
        """Third Y-derivative."""
        return self._compute_dy3()

    @cached_property
    def dydx2(self) -> torch.Tensor:
        """X-X-Y-derivative."""
        return self._compute_dydx2()

    @cached_property
    def dxdy2(self) -> torch.Tensor:
        """Y-Y-X-derivative."""
        return self._compute_dxdy2()

    def __init__(self, gaussian_support: GaussianSupport) -> None:
        """Instantiate the class.

        Args:
            gaussian_support (GaussianSupport): Gaussian support.
        """
        self.gs = gaussian_support
        self.gs_s = SumGaussianSupports(gaussian_support)

    def _compute(self) -> torch.Tensor:
        return torch.einsum("cxy,xy->cxy", self.gs.field, 1 / self.gs_s.field)

    def _compute_dx(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dx, self.gs_s.field)
            - torch.einsum("cxy,xy->cxy", self.gs.field, self.gs_s.dx)
        ) / self.gs_s.field.pow(2)

    def _compute_dy(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dy, self.gs_s.field)
            - torch.einsum("cxy,xy->cxy", self.gs.field, self.gs_s.dy)
        ) / self.gs_s.field.pow(2)

    def _compute_dx2(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dx2, self.gs_s.field.pow(2))
            - torch.einsum(
                "cxy,xy->cxy", self.gs.field, self.gs_s.dx2 * self.gs_s.field
            )
            - 2
            * torch.einsum(
                "cxy,xy->cxy", self.gs.dx, self.gs_s.dx * self.gs_s.field
            )
            + 2
            * torch.einsum("cxy,xy->cxy", self.gs.field, self.gs_s.dx.pow(2))
        ) / self.gs_s.field.pow(3)

    def _compute_dy2(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dy2, self.gs_s.field.pow(2))
            - torch.einsum(
                "cxy,xy->cxy", self.gs.field, self.gs_s.dy2 * self.gs_s.field
            )
            - 2
            * torch.einsum(
                "cxy,xy->cxy", self.gs.dy, self.gs_s.dy * self.gs_s.field
            )
            + 2
            * torch.einsum("cxy,xy->cxy", self.gs.field, self.gs_s.dy.pow(2))
        ) / self.gs_s.field.pow(3)

    def _compute_dydx(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dydx, self.gs_s.field.pow(2))
            - torch.einsum(
                "cxy,xy->cxy", self.gs.dx, self.gs_s.dy * self.gs_s.field
            )
            - torch.einsum(
                "cxy,xy->cxy", self.gs.dy, self.gs_s.dx * self.gs_s.field
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                -self.gs_s.dydx * self.gs_s.field
                + 2 * self.gs_s.dx * self.gs_s.dy,
            )
        ) / self.gs_s.field.pow(3)

    def _compute_dxdy(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dxdy, self.gs_s.field.pow(2))
            - torch.einsum(
                "cxy,xy->cxy", self.gs.dy, self.gs_s.dx * self.gs_s.field
            )
            - torch.einsum(
                "cxy,xy->cxy", self.gs.dx, self.gs_s.dy * self.gs_s.field
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                -self.gs_s.dxdy * self.gs_s.field
                + 2 * self.gs_s.dy * self.gs_s.dx,
            )
        ) / self.gs_s.field.pow(3)

    def _compute_dx3(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dx3, self.gs_s.field.pow(3))
            - 3
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dx2,
                self.gs_s.dx * self.gs_s.field.pow(2),
            )
            - 3
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dx,
                self.gs_s.dx2 * self.gs_s.field.pow(2)
                - 2 * self.gs_s.dx.pow(2) * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                6 * self.gs_s.dx2 * self.gs_s.dx * self.gs_s.field
                - self.gs_s.dx3 * self.gs_s.field.pow(2)
                - 6 * self.gs_s.dx.pow(3),
            )
        ) / self.gs_s.field.pow(4)

    def _compute_dydx2(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dydx2, self.gs_s.field.pow(3))
            - torch.einsum(
                "cxy,xy->cxy",
                self.gs.dx2,
                self.gs_s.dy * self.gs_s.field.pow(2),
            )
            - 2
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dydx,
                self.gs_s.dx * self.gs_s.field.pow(2),
            )
            + 2
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dx,
                -self.gs_s.dydx * self.gs_s.field
                + 2 * self.gs_s.dx * self.gs_s.dy * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.dy,
                -self.gs_s.dx2 * self.gs_s.field
                + 2 * self.gs_s.dx.pow(2) * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                -self.gs_s.dydx2 * self.gs_s.field
                + 2 * self.gs_s.dy * self.gs_s.dx2 * self.gs_s.field
                + 4 * self.gs_s.dydx * self.gs_s.dx * self.gs_s.field
                - 6 * self.gs_s.dx.pow(2) * self.gs_s.dy,
            )
        ) / self.gs_s.field.pow(4)

    def _compute_dy3(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dy3, self.gs_s.field.pow(3))
            - 3
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dy2,
                self.gs_s.dy * self.gs_s.field.pow(2),
            )
            - 3
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dy,
                self.gs_s.dy2 * self.gs_s.field.pow(2)
                - 2 * self.gs_s.dy.pow(2) * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                6 * self.gs_s.dy2 * self.gs_s.dy * self.gs_s.field
                - self.gs_s.dy3 * self.gs_s.field.pow(2)
                - 6 * self.gs_s.dy.pow(3),
            )
        ) / self.gs_s.field.pow(4)

    def _compute_dxdy2(self) -> torch.Tensor:
        return (
            torch.einsum("cxy,xy->cxy", self.gs.dxdy2, self.gs_s.field.pow(3))
            - torch.einsum(
                "cxy,xy->cxy",
                self.gs.dy2,
                self.gs_s.dx * self.gs_s.field.pow(2),
            )
            - 2
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dxdy,
                self.gs_s.dy * self.gs_s.field.pow(2),
            )
            + 2
            * torch.einsum(
                "cxy,xy->cxy",
                self.gs.dy,
                -self.gs_s.dxdy * self.gs_s.field
                + 2 * self.gs_s.dy * self.gs_s.dx * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.dx,
                -self.gs_s.dy2 * self.gs_s.field
                + 2 * self.gs_s.dy.pow(2) * self.gs_s.field,
            )
            + torch.einsum(
                "cxy,xy->cxy",
                self.gs.field,
                -self.gs_s.dxdy2 * self.gs_s.field
                + 2 * self.gs_s.dx * self.gs_s.dy2 * self.gs_s.field
                + 4 * self.gs_s.dxdy * self.gs_s.dy * self.gs_s.field
                - 6 * self.gs_s.dy.pow(2) * self.gs_s.dx,
            )
        ) / self.gs_s.field.pow(4)
