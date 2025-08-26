"""Non-homogenous boundary conditions."""

import torch

from qgsw.solver.finite_diff import laplacian1D


class BilinearExtendedBoundary:
    """Boundary extrapolation using bilinear functions."""

    @staticmethod
    def compute(
        x: torch.Tensor,
        y: torch.Tensor,
        ut: torch.Tensor,
        ub: torch.Tensor,
        ul: torch.Tensor,
        ur: torch.Tensor,
    ) -> torch.Tensor:
        """Extrapolate boundary values on rectangular domain to the interior.

        Approach: find a bilinear function that matches values at corners
            and interpolate residual boundary data using linear function
            of the complementary coordinate.

        Args:
            x (torch.Tensor): 1D x coordinates.
                └── (nx,)-shaped
            y (torch.Tensor): 1D y coordinates.
                └── (ny,)-shaped
            ut (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            ub (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            ul (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            ur (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped

        Returns:
            torch.Tensor: Bilinear-extended boundary field
                └── (..., nl, nx, ny)-shaped
        """
        ## Compute grid parameters
        a, b, c, d = x[0], x[-1], y[0], y[-1]
        xi = (x - a) / (b - a)  # (nx,)-shaped
        eta = (y - c) / (d - c)  # (ny,)-shaped
        xx, yy = torch.meshgrid(xi, eta, indexing="ij")
        # Reshape
        xx = xx[None, :, :]  # (1, nx, ny)-shaped
        yy = yy[None, :, :]  # (1, nx, ny)-shaped
        eta = eta[None, None, :]  # (1, 1, ny)-shaped
        xi = xi[None, :, None]  # (1, nx, 1)-shaped
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
        ula = ul[..., :, None, :] - u_bl * (1 - eta) - u_tl * eta
        ura = ur[..., :, None, :] - u_br * (1 - eta) - u_tr * eta
        # uta and uba: (..., nl, nx, 1)-shaped
        uba = ub[..., :, :, None] - u_bl * (1 - xi) - u_br * xi
        uta = ut[..., :, :, None] - u_tl * (1 - xi) - u_tr * xi
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

    @staticmethod
    def compute_laplacian(
        x: torch.Tensor,
        y: torch.Tensor,
        ut: torch.Tensor,
        ub: torch.Tensor,
        ul: torch.Tensor,
        ur: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the laplacian of the bilinear-extended boundary field.

        Δu = (1-x)∂²_y ul + x∂²_y ur + (1-y)∂²_x ub + y∂²_x ut

        Args:
            x (torch.Tensor): 1D x coordinates.
                └── (nx,)-shaped
            y (torch.Tensor): 1D y coordinates.
                └── (ny,)-shaped
            ut (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            ub (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            ul (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            ur (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped

        Returns:
            torch.Tensor: Laplacian of the bilinear-extended field obtained
            using `boundary_extension_bilinear`.
                └── (..., nl, nx, ny)-shaped
        """
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        a, b, c, d = x[0], x[-1], y[0], y[-1]
        xi_in, eta_in = (x[1:-1] - a) / (b - a), (y[1:-1] - c) / (d - c)
        xx_in, yy_in = torch.meshgrid(xi_in, eta_in, indexing="ij")
        xx_in = xx_in[None, :, :]  # (1, nx, ny)-shaped
        yy_in = yy_in[None, :, :]  # (1, nx, ny)-shaped
        eta_in = eta_in[None, None, :]  # (1, 1, ny)-shaped
        xi_in = xi_in[None, :, None]  # (1, nx, 1)-shaped
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
