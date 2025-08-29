"""Solvers for Helmholtz equations with Dirichlet boundary conditions."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import specs
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.solver.boundary_conditions.interpolation import (
    BilinearExtendedBoundary,
)
from qgsw.solver.helmholtz import compute_laplace_dstI, solve_helmholtz_dstI
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import defaults

if TYPE_CHECKING:
    from qgsw.solver.boundary_conditions.base import Boundaries


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
            return self._helmholtz_dstI.to(
                **defaults.get(device=device, dtype=dtype)
            )
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

    def _compute_homsol(
        self,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cst = torch.ones(
            (1, nl, nx + 1, ny + 1),
            **defaults.get(device=device, dtype=dtype),
        )
        helmholtz_dstI = self._compute_helmholtz_dstI(  # noqa: N806
            nx, ny, **defaults.get(device=device, dtype=dtype)
        )
        sol = solve_helmholtz_dstI(
            cst[..., 1:-1, 1:-1],
            helmholtz_dstI,
        )
        self._homsol = cst + sol * self._f0**2 * self._lambd
        homsol_i = points_to_surfaces(self._homsol)
        self._homsol_mean = homsol_i.mean((-1, -2), keepdim=True)
        return self._homsol, self._homsol_mean

    def _correct_sf_for_mass_conservation(
        self,
        sf_modes: torch.Tensor,
    ) -> torch.Tensor:
        """Corrects the stream function to ensure mass conservation.

        Args:
            sf_modes (torch.Tensor): Stream function modes
                └── (..., nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Corrected stream function modes
                └── (..., nl, nx+1, ny+1)-shaped
        """
        nl, nx, ny = sf_modes.shape[-3:]
        homsol, homsol_mean = self._compute_homsol(
            nl, nx - 1, ny - 1, **specs.from_tensor(sf_modes)
        )
        sf_modes_i = points_to_surfaces(sf_modes)
        sf_modes_i_mean = sf_modes_i.mean((-1, -2), keepdim=True)
        alpha = -sf_modes_i_mean / homsol_mean
        sf_modes += alpha * homsol
        return sf_modes

    def compute_stream_function(
        self,
        pv: torch.Tensor,
        *,
        ensure_mass_conservation: bool = True,
    ) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Homogeneous boundary conditions are assumed.

        Args:
            pv (torch.Tensor): Potential vorticity.
                └── (..., nl, nx, ny)-shaped
            ensure_mass_conservation (bool, optional): Whether to ensure mass
                conservation.

        Returns:
            torch.Tensor: Stream function
                └── (..., nl, nx+1, ny+1)-shaped
        """
        _, nx, ny = pv.shape[-3:]
        pv_i = points_to_surfaces(pv)

        rhs = torch.einsum("lm,...mxy->...lxy", self._Cl2m, pv_i)
        helmholtz_dstI = self._compute_helmholtz_dstI(  # noqa: N806
            nx,
            ny,
            **specs.from_tensor(pv),
        )
        sf_modes = solve_helmholtz_dstI(rhs, helmholtz_dstI)
        if ensure_mass_conservation:
            sf_modes = self._correct_sf_for_mass_conservation(sf_modes)
        return torch.einsum("ml,...lxy->...mxy", self._Cm2l, sf_modes)


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
            A (torch.Tensor): Stretching matrix
                └── (nl, nl)-shaped
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
        """
        super().__init__(A, f0, dx, dy)
        self._homogeneous_solver = HomogeneousPVInversion(A, f0, dx, dy)
        self._boundary = None

    def set_boundaries(self, boundaries: Boundaries) -> None:
        """Set the boundary values.

        Args:
            boundaries (Boundaries): Boundary conditions.
        """
        self._boundary = BilinearExtendedBoundary(boundaries)

    def set_boundaries_from_tensors(
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
        self._boundary = BilinearExtendedBoundary.from_tensors(ut, ub, ul, ur)

    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Boundary conditions are set to match self._boundary.

        Args:
            pv (torch.Tensor): Potential vorticity
                └── (..., nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function
                └── (..., nl, nx+1, ny+1)-shaped
        """
        sf_boundary = self._boundary.compute()

        pv_b = self._compute_pv_boundary(sf_boundary)
        pv_tot = pv - pv_b
        sf_homogeneous = self._homogeneous_solver.compute_stream_function(
            pv_tot, ensure_mass_conservation=False
        )
        return sf_homogeneous + sf_boundary

    def _compute_pv_boundary(self, sf_boundary: torch.Tensor) -> torch.Tensor:
        laplacian_boundary = self._boundary.compute_laplacian(
            self._dx, self._dy
        )
        return points_to_surfaces(
            F.pad(laplacian_boundary, (1, 1, 1, 1))
            - self._f0**2
            * torch.einsum("lm,...mxy->...lxy", self._A, sf_boundary)
        )
