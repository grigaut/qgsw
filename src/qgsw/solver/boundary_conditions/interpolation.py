"""Non-homogenous boundary conditions."""

import torch

from qgsw import specs
from qgsw.solver.finite_diff import laplacian1D


class BilinearExtendedBoundary:
    """Boundary extrapolation using bilinear functions."""

    def __init__(
        self,
        ut: torch.Tensor,
        ub: torch.Tensor,
        ul: torch.Tensor,
        ur: torch.Tensor,
    ) -> None:
        """Instantiate the boundary condition.

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
        self._ut = ut
        self._ub = ub
        self._ul = ul
        self._ur = ur
        self._compute_grids(ut, ul)

    def _compute_grids(self, ut: torch.Tensor, ul: torch.Tensor) -> None:
        """Compute the grids.

        Args:
            ut (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            ul (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
        """
        nx, ny = ut.shape[-1], ul.shape[-1]
        self._x = torch.linspace(0, 1, nx, **specs.from_tensor(ut))
        self._y = torch.linspace(0, 1, ny, **specs.from_tensor(ut))

    def compute(
        self,
    ) -> torch.Tensor:
        """Extrapolate boundary values on rectangular domain to the interior.

        Approach: find a bilinear function that matches values at corners
            and interpolate residual boundary data using linear function
            of the complementary coordinate.

        Returns:
            torch.Tensor: Bilinear-extended boundary field
                └── (..., nl, nx, ny)-shaped
        """
        ## Compute grid parameters
        xx, yy = torch.meshgrid(self._x, self._y, indexing="ij")
        # Reshape
        xx = xx[None, :, :]  # (1, nx, ny)-shaped
        yy = yy[None, :, :]  # (1, nx, ny)-shaped
        x = self._x[None, :, None]  # (1, nx, 1)-shaped
        y = self._y[None, None, :]  # (1, 1, ny)-shaped
        ## Variables
        ut = self._ut
        ub = self._ub
        ul = self._ul
        ur = self._ur
        ## Compute corner values
        u_bl = (ub[..., :, 0] + ul[..., :, 0]) / 2  # (..., nl)-shaped
        u_br = (ub[..., :, -1] + ur[..., :, 0]) / 2  # (..., nl)-shaped
        u_tl = (ut[..., :, 0] + ul[..., :, -1]) / 2  # (..., nl)-shaped
        u_tr = (ut[..., :, -1] + ur[..., :, -1]) / 2  # (..., nl)-shaped
        # Reshape
        u_bl = u_bl[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_br = u_br[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_tl = u_tl[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_tr = u_tr[..., :, None, None]  # (..., nl, 1, 1)-shaped
        ## Boundary values that are zero at corners
        # ula and ura: (..., nl, 1, ny)-shaped
        ula = ul[..., :, None, :] - u_bl * (1 - y) - u_tl * y
        ura = ur[..., :, None, :] - u_br * (1 - y) - u_tr * y
        # uta and uba: (..., nl, nx, 1)-shaped
        uba = ub[..., :, :, None] - u_bl * (1 - x) - u_br * x
        uta = ut[..., :, :, None] - u_tl * (1 - x) - u_tr * x
        # Sum linear extension of zero-ed boundary values + corner contribution
        # Empty corner contribution
        u_empty_corner = ula * (1 - xx) + ura * xx + uba * (1 - yy) + uta * yy
        # Corner contribution
        u_corners = (
            u_bl * (1 - xx) * (1 - yy)
            + u_br * xx * (1 - yy)
            + u_tl * (1 - xx) * yy
            + u_tr * xx * yy
        )
        return u_empty_corner + u_corners

    def compute_laplacian(
        self,
        dx: float,
        dy: float,
    ) -> torch.Tensor:
        """Compute the laplacian of the bilinear-extended boundary field.

        Δu = (1-x)∂²_y ul + x∂²_y ur + (1-y)∂²_x ub + y∂²_x ut

        Args:
            dx (float): Infinitesimal distance in the x direction.
            dy (float): Infinitesimal distance in the y direction.

        Returns:
            torch.Tensor: Laplacian of the bilinear-extended field obtained
            using `boundary_extension_bilinear`.
                └── (..., nl, nx-2, ny-2)-shaped
        """
        # Compute grid
        x_in = self._x[1:-1]
        y_in = self._y[1:-1]
        xx_in, yy_in = torch.meshgrid(x_in, y_in, indexing="ij")
        xx_in = xx_in[None, :, :]  # (1, nx, ny)-shaped
        yy_in = yy_in[None, :, :]  # (1, nx, ny)-shaped
        ## Variables
        ut = self._ut
        ub = self._ub
        ul = self._ul
        ur = self._ur
        # 1D laplacian of x-wise boundary conditions
        lap_ub = laplacian1D(ub, dx)
        lap_ut = laplacian1D(ut, dx)
        # 1D laplacian of y-wise boundary conditions
        lap_ul = laplacian1D(ul, dy)
        lap_ur = laplacian1D(ur, dy)
        # Reshape
        lap_ub = lap_ub[..., :, :, None]  # (..., nl, nx, 1)-shaped
        lap_ut = lap_ut[..., :, :, None]  # (..., nl, nx, 1)-shaped
        lap_ul = lap_ul[..., :, None, :]  # (..., nl, 1, ny)-shaped
        lap_ur = lap_ur[..., :, None, :]  # (..., nl, 1, ny)-shaped

        return (
            (1 - xx_in) * lap_ul
            + xx_in * lap_ur
            + (1 - yy_in) * lap_ub
            + yy_in * lap_ut
        )  # (..., nl, nx, ny)-shaped
