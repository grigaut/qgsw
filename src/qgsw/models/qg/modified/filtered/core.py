"""Modified QG model with filtered top layer."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.filters.high_pass import (
    SpectralGaussianHighPass2D,
)
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.modified.collinear_sublayer.core import QGAlpha
from qgsw.models.qg.modified.exceptions import UnsetAError, UnsetAlphaError
from qgsw.models.qg.projectors.core import QGProjector
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.specs import DEVICE
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.fields.variables.uvh import UVH
    from qgsw.filters.base import _Filter
    from qgsw.masks import Masks
    from qgsw.physics.coriolis.beta_plane import BetaPlane


@with_shapes(g_prime=(2,))
def compute_g_tilde(g_prime: torch.Tensor) -> torch.Tensor:
    """Compute g_tilde = g_1 g_2 / (g_1 + g_2).

    Args:
        g_prime (torch.Tensor): Reduced gravity tensor.
            └── (2,) shaped

    Returns:
        torch.Tensor: g_tilde = g_1 g_2 / (g_1 + g_2)
            └── (1,) shaped
    """
    if g_prime.shape != (2,):
        msg = f"g' should be (2,)-shaped, not {g_prime.shape}."
        raise ValueError(msg)
    g1, g2 = g_prime
    return (g1 * g2 / (g1 + g2)).unsqueeze(0)


class QGCollinearFilteredSF(QGAlpha["QGCollinearFilteredProjector"]):
    """Modified QG Model implementing collinear pv behavior."""

    _type = "QGCollinearFilteredSF"
    _supported_layers_nb = 2

    @with_shapes(H=(2,), g_prime=(2,))
    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Collinear Sublayer Stream Function.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (2,) shaped
            g_prime (torch.Tensor): Reduced Gravity Tensor.
                └── (2,) shaped
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        verbose.display(
            msg=f"Creating {self.__class__.__name__} model...",
            trigger_level=1,
        )
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
        )
        self._g_tilde = compute_g_tilde(g_prime)
        self._space = keep_top_layer(self._space)
        self._compute_coriolis(self._space.omega.remove_z_h())
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._set_state()
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H[:1],
            g_prime=self._g_tilde,
            beta_plane=beta_plane,
            optimize=optimize,
        )
        self.A = self.compute_A(
            H[:1],
            self._g_tilde,
        )
        self._set_projector()

    @QGAlpha.alpha.setter
    @with_shapes(alpha=(1,))
    def alpha(self, alpha: torch.Tensor) -> None:
        """Setter for alpha."""
        QGAlpha.alpha.fset(self, alpha)
        self._P.alpha = alpha

    def _set_projector(self) -> None:
        self._P = QGCollinearFilteredProjector(
            A=self.A,
            H=self.H,
            g_prime=self.g_prime,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )


class QGCollinearFilteredProjector(QGProjector):
    """QG projector for QGCollinearFilteredSF."""

    @with_shapes(
        A=(1, 1),
        H=(2, 1, 1),
        g_prime=(2, 1, 1),
    )
    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        space: SpaceDiscretization3D,
        f0: float,
        masks: Masks,
    ) -> None:
        """Instantiate the projector.

        Args:
            A (torch.Tensor): Stretching matrix.
                └── (1, 1)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (1, 1, 1)-shaped
            g_prime (torch.Tensor): Reduced gravity.
                └── (2, 1, 1)-shaped
            space (SpaceDiscretization3D): 3D space discretization.
            f0 (float): f0.
            masks (Masks): Masks.
        """
        self._filter = SpectralGaussianHighPass2D(1)
        self._g_prime = g_prime
        super().__init__(A, H[:1], space, f0, masks)
        self._g_tilde = compute_g_tilde(g_prime[..., 0, 0])

    @property
    def filter(self) -> SpectralGaussianHighPass2D:
        """Filter."""
        return self._filter

    @QGProjector.A.setter
    def A(self, A: torch.Tensor) -> None:  # noqa: N802, N803
        """Set the streching matrix."""
        self._A = A
        decomposition = compute_layers_to_mode_decomposition(A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
        with contextlib.suppress(UnsetAlphaError):
            self._set_helmholtz_solver(self.lambd, self.alpha, self._f0)

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        try:
            return self._alpha
        except AttributeError as e:
            raise UnsetAlphaError from e

    @alpha.setter
    @with_shapes(alpha=(1,))
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        with contextlib.suppress(UnsetAError):
            self._set_helmholtz_solver(self.lambd, self.alpha, self._f0)

    @classmethod
    @with_shapes(
        H=(1, 1, 1),
        g_prime=(2, 1, 1),
        g_tilde=(1,),
        alpha=(1,),
    )
    def Q(  # noqa: N802
        cls,
        uvh: UVH,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        g_tilde: torch.Tensor,
        f0: float,
        ds: float,
        filt: _Filter,
        alpha: torch.Tensor,
        points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, 1, nx+1, ny)-shaped
                ├── v: (n_ens, 1, nx, ny+1)-shaped
                └── h: (n_ens, 1, nx, ny)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (1, 1, 1)-shaped.
            g_prime (torch.Tensor): Reduced gravity.
                └── (2, 1, 1)-shaped.
            g_tilde (torch.Tensor): Equivalent reduced gravity.
                └── (1, )-shaped.
            f0 (float): f0.
            ds (float): ds.
            filt (_Filter): Filter.
            alpha (torch.Tensor): Collinearity coefficient.
                └── (1, )-shaped.
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface interpolation function.

        Returns:
            torch.Tensor: Physical Pressure * f0.
                └── (n_ens, nl, nx-1, ny-1)-shaped.
        """
        pv = QGProjector.Q(uvh, H, f0, ds, points_to_surfaces)
        source_term = QGCollinearFilteredProjector.compute_source_term(
            uvh=uvh,
            filt=filt,
            H1=H[..., 0, 0],
            g2=g_prime[1:, 0, 0],
            g_tilde=g_tilde,
            alpha=alpha,
            f0=f0,
            ds=ds,
            points_to_surface=points_to_surfaces,
        )
        return pv + source_term

    def _Q(self, uvh: UVH) -> torch.Tensor:  # noqa: N802
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical Pressure * f0.
                └── (n_ens, nl, nx-1, ny-1)-shaped.
        """
        return self.Q(
            uvh=uvh,
            H=self.H,
            g_prime=self._g_prime,
            g_tilde=self._g_tilde,
            f0=self._f0,
            ds=self._space.ds,
            filt=self._filter,
            alpha=self.alpha,
            points_to_surfaces=self._points_to_surface,
        )

    @classmethod
    def compute_source_term_factor(
        cls,
        alpha: torch.Tensor,
        H1: torch.Tensor,  # noqa: N803
        g2: torch.Tensor,
        f0: float,
    ) -> torch.Tensor:
        """Compute source term multiplicative factor.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.
                └── (1, )-shaped.
            H1 (torch.Tensor): Top layer reference depth.
                └── (1, )-shaped.
            g2 (torch.Tensor): Reduced gravity in the second layer.
                └── (1, )-shaped.
            f0 (float): f0.

        Returns:
            torch.Tensor: f_0²α/H1/g2
        """
        return f0**2 * alpha / H1 / g2

    @classmethod
    @with_shapes(
        alpha=(1,),
        H1=(1,),
        g2=(1,),
        g_tilde=(1,),
    )
    def compute_source_term(
        cls,
        uvh: UVH,
        filt: _Filter,
        alpha: torch.Tensor,
        H1: torch.Tensor,  # noqa: N803
        g2: torch.Tensor,
        g_tilde: torch.Tensor,
        f0: float,
        ds: float,
        points_to_surface: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute source term.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, 1, nx+1, ny)-shaped
                ├── v: (n_ens, 1, nx, ny+1)-shaped
                └── h: (n_ens, 1, nx, ny)-shaped
            filt (_Filter): Filter.
            alpha (torch.Tensor): Collinearity coefficient.
            H1 (torch.Tensor): Top layer depth.
                └── (1, )-shaped.
            g2 (torch.Tensor): Reduced gravity in the bottom layer.
                └── (1, )-shaped.
            g_tilde (torch.Tensor): Equivalent reduced gravity.
                └── (1, )-shaped.
            f0 (float): f0.
            ds (float): ds.
            points_to_surface (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface interpolation function.

        Returns:
            torch.Tensor: Source term: f_0²αg̃/H1/g2/ds (F^s)⁻¹{K F{h}}.
        """
        h_top_i = points_to_surface(uvh.h[0, 0])
        h_filt = filt(h_top_i).unsqueeze(0).unsqueeze(0)
        h_to_psi = g_tilde * h_filt
        factor = cls.compute_source_term_factor(alpha, H1, g2, f0)
        return factor * h_to_psi / ds

    def _set_helmholtz_solver(
        self,
        lambd: torch.Tensor,
        alpha: torch.Tensor,
        f0: float,
    ) -> None:
        """Set the Helmholtz Solver.

        Args:
            lambd (torch.Tensor): Matrix A's eigenvalues.
                └── (1, 1, 1, 1)-shaped.
            alpha (torch.Tensor): Collinearity coefficient.
                └── (1, 1, 1, 1)-shaped.
            f0 (float): f0.
        """
        # For Helmholtz equations
        nl, nx, ny = self._space.nl, self._space.nx, self._space.ny
        dx, dy = self._space.dx, self._space.dy
        laplace_dstI = (  # noqa: N806
            compute_laplace_dstI(
                nx,
                ny,
                dx,
                dy,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Compute "(∆ - (f_0)² Λ)" in Fourier Space
        helmholtz_dstI = laplace_dstI - f0**2 * lambd  # noqa: N806

        factor = self.compute_source_term_factor(
            alpha,
            self.H[0, 0, 0],
            self._g_prime[1, 0, 0],
            f0,
        )

        K = self.filter.compute_kernel(  # noqa: N806
            self.filter.sigma,
            nx=nx,
            ny=ny,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

        # Compute "(∆ - (f_0)² Λ + f_0²αK/H1/g2)" in Fourier Space
        helmholtz_dstI += factor * self._points_to_surface(K)  # noqa: N806

        # Constant Omega grid
        cst_wgrid = torch.ones(
            (1, nl, nx + 1, ny + 1),
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        if len(self._masks.psi_irrbound_xids) > 0:
            # Handle Non rectangular geometry
            cap_matrices = compute_capacitance_matrices(
                helmholtz_dstI,
                self._masks.psi_irrbound_xids,
                self._masks.psi_irrbound_yids,
            )
            sol_wgrid = solve_helmholtz_dstI_cmm(
                (cst_wgrid * self._masks.psi)[..., 1:-1, 1:-1],
                helmholtz_dstI,
                cap_matrices,
                self._masks.psi_irrbound_xids,
                self._masks.psi_irrbound_yids,
                self._masks.psi,
            )

            def compute_p_modes(helmholtz_rhs: torch.Tensor) -> torch.Tensor:
                return solve_helmholtz_dstI_cmm(
                    helmholtz_rhs * self._masks.psi[..., 1:-1, 1:-1],
                    helmholtz_dstI,
                    cap_matrices,
                    self._masks.psi_irrbound_xids,
                    self._masks.psi_irrbound_yids,
                    self._masks.psi,
                )
        else:
            sol_wgrid = solve_helmholtz_dstI(
                cst_wgrid[..., 1:-1, 1:-1],
                helmholtz_dstI,
            )

            def compute_p_modes(helmholtz_rhs: torch.Tensor) -> torch.Tensor:
                return solve_helmholtz_dstI(helmholtz_rhs, helmholtz_dstI)

        self._compute_p_modes = compute_p_modes
        # Compute homogenous solution
        self.homsol_wgrid = cst_wgrid + sol_wgrid * f0**2 * lambd
        self.homsol_wgrid_mean = self.homsol_wgrid.mean((-1, -2), keepdim=True)
