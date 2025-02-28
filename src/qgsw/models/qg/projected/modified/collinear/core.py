"""Modified QG Model with Collinear Sublayer Behavior."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, TypeVar

import torch

from qgsw import verbose
from qgsw.fields.variables.prognostic_tuples import UVH, UVHTAlpha
from qgsw.fields.variables.state import StateUVHAlpha
from qgsw.models.core.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dstI,
    solve_helmholtz_dstI,
    solve_helmholtz_dstI_cmm,
)
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.projected.core import QGCore
from qgsw.models.qg.projected.modified.collinear.variable_set import (
    QGCollinearSFVariableSet,
)
from qgsw.models.qg.projected.modified.exceptions import (
    UnsetAError,
    UnsetAlphaError,
)
from qgsw.models.qg.projected.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.projected.projectors.core import QGProjector
from qgsw.models.qg.stretching_matrix import (
    compute_A,
    compute_layers_to_mode_decomposition,
)
from qgsw.models.sw.core import SWCollinearSublayer
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.specs import defaults
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.masks import Masks
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )

Projector = TypeVar("Projector", bound=QGProjector)


class QGAlpha(QGCore[UVHTAlpha, StateUVHAlpha, Projector]):
    """Collinear QG Model."""

    _supported_layers_nb: int
    _A: torch.Tensor

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)

    def _set_state(self) -> None:
        """Set the state."""
        self._state = StateUVHAlpha.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self._io = IO(
            t=self._state.t,
            u=self._state.u,
            v=self._state.v,
            h=self._state.h,
            alpha=self._state.alpha,
        )

    def _set_H(self, h: torch.Tensor) -> None:  # noqa: N802
        """Perform additional validation over H.

        Args:
            h (torch.Tensor): Layers thickness.
                └── h: (nl, 1, 1)-shaped

        Raises:
            ValueError: if H is not constant in space
        """
        if self.space.nl != self._supported_layers_nb:
            msg = (
                f"QGAlpha can only support{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        super()._set_H(h)


class QGCollinearSF(QGAlpha["QGCollinearProjector"]):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _type = ModelName.QG_COLLINEAR_SF

    _supported_layers_nb: int = 2
    _coefficient_set = False
    _core: SWCollinearSublayer

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
            H=H,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )
        self.A = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])
        self._set_projector()

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness.

        └── (1, 1, 1) shaped
        """
        return self._H[:1, ...]

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity.

        └── (1, 1, 1) shaped
        """
        return self._g_prime[:1, ...]

    @QGAlpha.alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        """Alpha setter."""
        QGAlpha.alpha.fset(self, alpha)
        self._core.alpha = alpha
        self.P.alpha = alpha
        self._create_diagnostic_vars(self._state)

    def _init_core_model(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        optimize: bool,  # noqa: FBT001
    ) -> SWCollinearSublayer:
        """Initialize the core Shallow Water model.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor.
                └── (2,) shaped
            g_prime (torch.Tensor): Reduced Gravity Tensor.
                └── (2,) shaped
            beta_plane (Beta_Plane): Beta plane.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.

        Returns:
            SW: Core model (One layer).
        """
        return SWCollinearSublayer(
            space_2d=space_2d,
            H=H,  # Only consider top layer
            g_prime=g_prime,  # Only consider top layer
            beta_plane=beta_plane,
            optimize=optimize,
        )

    def compute_A(  # noqa: N802
        self,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Compute new Stretching operator.

        Ã = (1/ρ_1)[[(1/H_1)*(1/g_1 + (1 - α)/g_2)]]

        Args:
            H (torch.Tensor): Layers reference height.
                └── (2,) shaped
            g_prime (torch.Tensor): Reduced gravity values.
                └── (2,) shaped

        Returns:
            torch.Tensor: Stretching Operator
        """
        return compute_A(
            H=H[:1],
            g_prime=compute_g_tilde(g_prime),
            dtype=self.dtype,
            device=self.device.get(),
        )

    def _set_projector(self) -> None:
        self._P = QGCollinearProjector(
            self.A,
            self._H,
            g_prime=self._g_prime,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        return super().step()

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,
        physics: PhysicsConfig,
        model: ModelConfig,
    ) -> dict[str, DiagnosticVariable]:
        """Create variable set.

        Args:
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
            model (ModelConfig): Model configuaration.

        Returns:
            dict[str, DiagnosticVariable]: Variables dictionnary.
        """
        return QGCollinearSFVariableSet.get_variable_set(space, physics, model)


class QGCollinearProjector(QGProjector):
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
        self._g_prime = g_prime
        super().__init__(A, H[:1], space, f0, masks)

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
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        with contextlib.suppress(UnsetAError):
            self._set_helmholtz_solver(self.lambd, self.alpha, self._f0)

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
                **defaults.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # Compute "(∆ - (f_0)² Λ)" in Fourier Space
        helmholtz_dstI = laplace_dstI - f0**2 * lambd  # noqa: N806

        # Compute "(∆ - (f_0)² Λ + f_0²α/H1/g2)" in Fourier Space
        source_term = (
            f0**2
            * self._points_to_surface(alpha)
            / self.H[0, 0, 0]
            / self._g_prime[1, 0, 0]
        )
        helmholtz_dstI += source_term  # noqa: N806

        # Constant Omega grid
        cst_wgrid = torch.ones(
            (1, nl, nx + 1, ny + 1),
            **defaults.get(),
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

    @classmethod
    @with_shapes(
        g2=(1,),
        H=(1, 1, 1),
    )
    def G(  # noqa: N802
        cls,
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
        ds: float,
        f0: float,
        g2: torch.Tensor,
        alpha: torch.Tensor,
        points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
        p_i: torch.Tensor | None = None,
    ) -> UVH:
        """Geostrophic operator.

        Args:
            p (torch.float):Pressure.
                └── (n_ens, nl, nx+1, ny+1)-shaped
            A (torch.Tensor): Stretching matrix.
                └── (nl,nl)-shaped.
            H (torch.Tensor): Layers reference thickness.
                └── (nl, 1, 1)-shaped.
            dx (float): dx.
            dy (float): dy.
            ds (float): ds.
            f0 (float): f0.
            g2 (torch.Tensor): Reduced gravity in the second layer.
                └── (1,)-shaped
            alpha (torch.Tensor): Collinearity coefficient.
                └── (1,)-shaped
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface function.
            p_i (torch.Tensor | None, optional): Interpolated pressure.
            Defaults to None.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Prognostic variables u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        p_i = points_to_surfaces(p) if p_i is None else p_i

        # geostrophic balance
        u = -torch.diff(p, dim=-1) / dy / f0 * dx
        v = torch.diff(p, dim=-2) / dx / f0 * dy
        # source_term = α A_{1,2} p
        source_term = alpha / H[0, 0, 0] / g2 * p_i
        # h = diag(H)(Ap-αA_{1,2}p)
        h = H * (torch.einsum("lm,...mxy->...lxy", A, p_i) - source_term) * ds

        return UVH(u, v, h)

    def _G(self, p: torch.Tensor, p_i: torch.Tensor | None) -> UVH:  # noqa: N802
        """Geostrophic operator.

        Args:
            p (torch.float):Pressure, (n_ens, nl, nx+1, ny+1)-shaped.
                └── (n_ens, nl, nx+1, ny+1)-shaped
            p_i (torch.Tensor | None): Interpolated pressure.
                └── (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: Prognostic variables u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return self.G(
            p=p,
            p_i=p_i,
            A=self._A,
            H=self._H,
            dx=self._space.dx,
            dy=self._space.dy,
            ds=self._space.ds,
            f0=self._f0,
            g2=self._g_prime[1:2, 0, 0],
            alpha=self.alpha,
            points_to_surfaces=self._points_to_surface,
        )
