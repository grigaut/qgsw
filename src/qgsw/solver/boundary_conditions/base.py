"""Base classes for boundary conditions."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass
from functools import cached_property

import torch


@dataclass(frozen=True)
class Boundaries:
    """Named tuple for boundary conditions.

    Args:
        top (torch.Tensor): Boundary condition at the top (y=y_max)
            └── (..., nl, ny)-shaped
        bottom (torch.Tensor): Boundary condition at the bottom (y=y_min)
            └── (..., nl, ny)-shaped
        left (torch.Tensor): Boundary condition on the left (x=x_min)
            └── (..., nl, nx)-shaped
        right (torch.Tensor): Boundary condition on the right (x=x_max)
            └── (..., nl, nx)-shaped
    """

    top: torch.Tensor
    bottom: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    @cached_property
    def nx(self) -> int:
        """Number of points in the x direction."""
        return self.top.shape[-1]

    @cached_property
    def ny(self) -> int:
        """Number of points in the y direction."""
        return self.left.shape[-1]

    def __add__(self, other: "Boundaries") -> "Boundaries":
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __radd__(self, other: "Boundaries") -> "Boundaries":
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __sub__(self, other: "Boundaries") -> "Boundaries":
        """Subtract two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top - other.top,
            bottom=self.bottom - other.bottom,
            left=self.left - other.left,
            right=self.right - other.right,
        )

    def __rmul__(self, scalar: float) -> "Boundaries":
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __mul__(self, scalar: float) -> "Boundaries":
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __truediv__(self, scalar: float) -> "Boundaries":
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top / scalar,
            bottom=self.bottom / scalar,
            left=self.left / scalar,
            right=self.right / scalar,
        )

    def __post_init__(self) -> None:
        """Post initialization method.

        Raises:
            ValueError: If the boundary shapes are not compatible.
            ValueError: If the boundary shapes are not compatible.
        """
        if self.bottom.shape != self.top.shape:
            msg = "Both bottom and top boundaries must have the same shape."
            raise ValueError(msg)
        if self.left.shape != self.right.shape:
            msg = "Both left and right boundaries must have the same shape."
            raise ValueError(msg)

    @classmethod
    def extract(
        cls, field: torch.Tensor, imin: int, imax: int, jmin: int, jmax: int
    ) -> Self:
        """Extract boundary conditions from a field.

        Args:
            field (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+1)-shaped
                ├── bottom: (..., imax-imin+1)-shaped
                ├── left: (..., jmax-jmin+1)-shaped
                └── right: (..., jmax-jmin+1)-shaped
        """
        return cls(
            top=field[..., imin : imax + 1, jmax],
            bottom=field[..., imin : imax + 1, jmin],
            left=field[..., imin, jmin : jmax + 1],
            right=field[..., imax, jmin : jmax + 1],
        )

    @classmethod
    def extract_sf(
        cls, sf: torch.Tensor, imin: int, imax: int, jmin: int, jmax: int
    ) -> Self:
        """Extract boundary conditions from a stream function field.

        Args:
            sf (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+1)-shaped
                ├── bottom: (..., imax-imin+1)-shaped
                ├── left: (..., jmax-jmin+1)-shaped
                └── right: (..., jmax-jmin+1)-shaped
        """
        return cls.extract(sf, imin, imax, jmin, jmax)

    @classmethod
    def extract_pv(
        cls, pv: torch.Tensor, imin: int, imax: int, jmin: int, jmax: int
    ) -> Self:
        """Extract boundary conditions from a potential vorticity field.

        Args:
            pv (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+2)-shaped
                ├── bottom: (..., imax-imin+2)-shaped
                ├── left: (..., jmax-jmin+2)-shaped
                └── right: (..., jmax-jmin+2)-shaped
        """
        return cls.extract(pv, imin - 1, imax, jmin - 1, jmax)


@dataclass(frozen=True)
class TimedBoundaries:
    """Named tuple for time-dependent boundary conditions."""

    time: float
    boundaries: Boundaries

    @classmethod
    def from_tensors(
        cls,
        time: float,
        top: torch.Tensor,
        bottom: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Self:
        """Instantiate the boundaries using tensors.

        Args:
            time (float): Time.
            top (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            bottom (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            left (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            right (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped

        Returns:
            Self: TimedBoundary.
        """
        return cls(
            time=time,
            boundaries=Boundaries(
                top=top, bottom=bottom, left=left, right=right
            ),
        )

    @classmethod
    def extract(
        cls,
        time: float,
        field: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> Self:
        """Extract boundary conditions from a field.

        Args:
            time (float): Time of the boundary conditions.
            field (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+1)-shaped
                ├── bottom: (..., imax-imin+1)-shaped
                ├── left: (..., jmax-jmin+1)-shaped
                └── right: (..., jmax-jmin+1)-shaped
        """
        return cls(
            time=time,
            boundaries=Boundaries.extract(field, imin, imax, jmin, jmax),
        )

    @classmethod
    def extract_sf(
        cls,
        time: float,
        sf: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> Self:
        """Extract boundary conditions from a stream function field.

        Args:
            time (float): Time of the boundary conditions.
            sf (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+1)-shaped
                ├── bottom: (..., imax-imin+1)-shaped
                ├── left: (..., jmax-jmin+1)-shaped
                └── right: (..., jmax-jmin+1)-shaped
        """
        return cls(
            time=time,
            boundaries=Boundaries.extract_sf(sf, imin, imax, jmin, jmax),
        )

    @classmethod
    def extract_pv(
        cls,
        time: float,
        pv: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> Self:
        """Extract boundary conditions from a potential vorticity field.

        Args:
            time (float): Time of the boundary conditions.
            pv (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+2)-shaped
                ├── bottom: (..., imax-imin+2)-shaped
                ├── left: (..., jmax-jmin+2)-shaped
                └── right: (..., jmax-jmin+2)-shaped
        """
        return cls(
            time=time,
            boundaries=Boundaries.extract_pv(pv, imin, imax, jmin, jmax),
        )
