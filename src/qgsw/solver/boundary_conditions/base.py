"""Base classes for boundary conditions."""

from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812


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

    def __add__(self, other: Boundaries) -> Boundaries:
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __radd__(self, other: Boundaries) -> Boundaries:
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __sub__(self, other: Boundaries) -> Boundaries:
        """Subtract two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top - other.top,
            bottom=self.bottom - other.bottom,
            left=self.left - other.left,
            right=self.right - other.right,
        )

    def __rmul__(self, scalar: float) -> Boundaries:
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __mul__(self, scalar: float) -> Boundaries:
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __truediv__(self, scalar: float) -> Boundaries:
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

    @overload
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int,
        inplace: Literal[False],
    ) -> torch.Tensor: ...
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int = 0,
        inplace: bool = True,
    ) -> torch.Tensor | None:
        """Add boundary to a given Tensor.

        Args:
            field (torch.Tensor): Field to add boundary to.
            offset (int, optional): Offset. If 0, the outer most border
                will be completed. Defaults to 0.
            inplace (bool, optional): Whether to perform operation inplace or
                not. Defaults to True.

        Returns:
            torch.Tensor | None: Tensor with boundary if inplace is True,
                otherwise nothing.
        """
        if not inplace:
            field = torch.clone(field)
        nx, ny = field.shape[-2:]
        lx = nx - offset * 2
        ly = ny - offset * 2
        if lx != self.bottom.shape[-1]:
            msg = "Mismatching size at top/bottom boundary."
            raise ValueError(msg)
        if ly != self.left.shape[-1]:
            msg = "Mismatching size at left/right boundary."
            raise ValueError(msg)
        field_left = field[..., offset, :].narrow(-1, offset, ly)
        field_left[:] = self.left
        field_right = field[..., -(offset + 1), :].narrow(-1, offset, ly)
        field_right[:] = self.right
        field_bottom = field[..., :, offset].narrow(-1, offset, lx)
        field_bottom[:] = self.bottom
        field_top = field[..., :, -(offset + 1)].narrow(-1, offset, lx)
        field_top[:] = self.top
        return field if not inplace else None

    def expand(self, field: torch.Tensor) -> torch.Tensor:
        """Expand a tensor by adding the boundary.

        Args:
            field (torch.Tensor): Tensor to expand.

        Returns:
            torch.Tensor: Expanded tensor.
        """
        return self.set_to(F.pad(field, (1, 1, 1, 1)), inplace=False)

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
        return cls(
            top=sf[..., imin : imax + 1, jmax],
            bottom=sf[..., imin : imax + 1, jmin],
            left=sf[..., imin, jmin : jmax + 1],
            right=sf[..., imax, jmin : jmax + 1],
        )

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
        return cls(
            top=pv[..., imin:imax, jmax - 1],
            bottom=pv[..., imin:imax, jmin],
            left=pv[..., imin, jmin:jmax],
            right=pv[..., imax - 1, jmin:jmax],
        )


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
