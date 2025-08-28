"""Solvers for Helmholtz equations with Dirichlet boundary conditions."""

from __future__ import annotations

from abc import abstractmethod

import torch

from qgsw import specs
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.solver.boundary_conditions import BilinearExtendedBoundary
from qgsw.solver.helmholtz import compute_laplace_dstI, solve_helmholtz_dstI
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import defaults


class BasePVInversion:
    """Base class for Helmoholtz solvers with Dirichlet boundary conditions."""

    _nx = None
    _ny = None

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
    ) -> None:
        """Base class for potential vorticity inversion.

        Args:
            A (torch.Tensor): Stretching matrix.
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
        """
        self._A = A
        self._nl = A.shape[0]
        self._f0 = f0
        self._dx = dx
        self._dy = dy

        Cm2l, lambd, Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        self._Cm2l = Cm2l
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
        self._Cl2m = Cl2m

    def _compute_helmholtz_dstI(  # noqa: N802
        self,
        nx: int,
        ny: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute the Helmholtz operator in spectral space.

        Args:
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            dtype (torch.dtype | None, optional): Data type of the tensor.
                Defaults to None.
            device (torch.device | None, optional): Device to create
            the tensor on. Defaults to None.

        Returns:
            torch.Tensor: Helmholtz operator in spectral space.
        """
        if (nx, ny) == (self._nx, self._ny):
            return self._helmholtz_dstI
        self._nx, self._ny = nx, ny
        laplacian = compute_laplace_dstI(
            nx,
            ny,
            self._dx,
            self._dy,
            **defaults.get(device=device, dtype=dtype),
        )
        laplacian = laplacian.unsqueeze(0).unsqueeze(0)
        self._helmholtz_dstI = laplacian - self._f0**2 * self._lambd
        return self._helmholtz_dstI

    @abstractmethod
    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Args:
            pv (torch.Tensor): Potential vorticity, shape (..., nl, nx, ny).

        Returns:
            torch.Tensor: Stream function, shape (..., nl, nx+1, ny+1).
        """


class HomogeneousPVInversion(BasePVInversion):
    """Homogeneous potential vorticity inversion."""

    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Homogeneous boundary conditions are assumed.

        Args:
            pv (torch.Tensor): Potential vorticity, shape (..., nl, nx, ny).

        Returns:
            torch.Tensor: Stream function, shape (..., nl, nx+1, ny+1).
        """
        nx, ny = pv.shape[-2:]
        pv_i = points_to_surfaces(pv)

        rhs = torch.einsum("lm,...mxy->...lxy", self._Cl2m, pv_i)
        helmholtz_dstI = self._compute_helmholtz_dstI(  # noqa: N806
            nx,
            ny,
            **specs.from_tensor(pv),
        )
        sf = solve_helmholtz_dstI(rhs, helmholtz_dstI)
        return torch.einsum("ml,...lxy->...mxy", self._Cm2l, sf)


class InhomogeneousPVInversion(BasePVInversion):
    """Inhomogeneous potential vorticity inversion."""

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
    ) -> None:
        """Base class for potential vorticity inversion.

        Args:
            A (torch.Tensor): Stretching matrix.
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
        """
        super().__init__(A, f0, dx, dy)
        self._homogeneous_solver = HomogeneousPVInversion(A, f0, dx, dy)
        self._boundary = None

    def set_boundaries(
        self,
        ut: torch.Tensor,
        ub: torch.Tensor,
        ul: torch.Tensor,
        ur: torch.Tensor,
    ) -> None:
        """Set the boundary values.

        Args:
            ut (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            ub (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            ul (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            ur (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped
        """
        self._boundary = BilinearExtendedBoundary(ut, ub, ul, ur)

    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Boundary conditions are set to match self._boundary.

        Args:
            pv (torch.Tensor): Potential vorticity, shape (..., nl, nx, ny).

        Returns:
            torch.Tensor: Stream function, shape (..., nl, nx+1, ny+1).
        """
        sf_boundary = self._boundary.compute()

        pv_b = self._compute_pv_boundary(sf_boundary)
        pv_tot = pv - pv_b
        sf_homogeneous = self._homogeneous_solver.compute_stream_function(
            pv_tot
        )
        return sf_homogeneous + sf_boundary

    def _compute_pv_boundary(self, sf_boundary: torch.Tensor) -> torch.Tensor:
        laplacian_boundary = self._boundary.compute_laplacian(
            self._dx, self._dy
        )
        return points_to_surfaces(
            torch.nn.functional.pad(
                laplacian_boundary,
                (1, 1, 1, 1),
            )
            - self._f0**2
            * torch.einsum("lm,...mxy->...lxy", self._A, sf_boundary)
        )
