"""Projector for Collinear models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.fields.variables.tuples import UVH
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.modified.collinear.stretching_matrix import (
    compute_A_12,
)
from qgsw.models.qg.uvh.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.synchronization.rescaling import interpolate_physical_variable
from qgsw.specs import defaults
from qgsw.utils.shape_checks import with_shapes

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from qgsw.masks import Masks
from qgsw.spatial.core.discretization import SpaceDiscretization3D

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig


class CollinearSFProjector(QGProjector):
    """QG Projector considering collinear stream function."""

    _MAX_ITERATIONS = 1000
    _ATOL = 1e-8  # default value for torch.isclose
    _RTOL = 1e-5  # default value for torch.isclose

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
            H (torch.Tensor): Layers reference thickness.
            g_prime (torch.Tensor): Reduced gravity in the layer 2.
            space (SpaceDiscretization3D): 3D space discretization.
            f0 (float): _description_
            masks (Masks): _description_
        """
        super().__init__(
            A=A,
            H=H,
            space=space,
            f0=f0,
            masks=masks,
        )
        self._g_prime = g_prime

    @property
    def alpha(self) -> torch.Tensor:
        """Alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha

    @QGProjector.A.setter
    @with_shapes(A=(1, 1))
    def A(self, A: torch.Tensor) -> None:  # noqa: N802, N803
        """Set the stretching matrix."""
        # A.shape = (1,1) -> Cm2L = [[1]], lambd = A_11,  Cl2M = [[1]]
        QGProjector.A.fset(self, A)

    def _set_helmholtz_solver(self, lambd: torch.Tensor, f0: float) -> None:
        """Set the Helmholtz Solver.

        Args:
            lambd (torch.Tensor): Matrix A's eigenvalues.
                └── (1, 1, 1, 1)-shaped.
            f0 (float): f0.
        """
        if len(self._masks.psi_irrbound_xids) > 0:
            # Handle Non rectangular geometry
            raise NotImplementedError
        super()._set_helmholtz_solver(lambd, f0)

    @classmethod
    @with_shapes(
        H=(2, 1, 1),
        A=(1, 1),
        g_prime=(2, 1, 1),
    )
    def G(  # noqa: N802
        cls,
        p: torch.Tensor,
        A: torch.Tensor,  # noqa: N803
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        dx: float,
        dy: float,
        ds: float,
        f0: float,
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
                └── (2, 1, 1)-shaped.
            g_prime (torch.Tensor): Reduced gravity.
                └── (2, 1, 1)-shaped.
            dx (float): dx.
            dy (float): dy.
            ds (float): ds.
            f0 (float): f0.
            alpha (torch.Tensor): Collinearity coeffciient.
                └── (nx, ny)-shaped.
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
        # h = diag(H)Ap
        A_12 = compute_A_12(H[:, 0, 0], g_prime[:, 0, 0])  # noqa: N806
        h = (
            H[0, 0, 0]
            * (torch.einsum("lm,...mxy->...lxy", A, p_i) + A_12 * alpha * p_i)
            * ds
        )

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
            g_prime=self._g_prime,
            dx=self._space.dx,
            dy=self._space.dy,
            ds=self._space.ds,
            f0=self._f0,
            alpha=self.alpha,
            points_to_surfaces=self._points_to_surface,
        )

    @classmethod
    @with_shapes(
        H=(2, 1, 1),
    )
    def Q(  # noqa: N802
        cls,
        uvh: UVH,
        H: torch.Tensor,  # noqa: N803
        f0: float,
        ds: float,
        points_to_surfaces: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """PV linear operator.

        Args:
            uvh (UVH): Prognostic u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
            H (torch.Tensor): Layers reference thickness.
                └── (nl, 1, 1)-shaped.
            f0 (float): f0.
            ds (float): ds.
            points_to_surfaces (Callable[[torch.Tensor], torch.Tensor]): Points
            to surface interpolation function.

        Returns:
            torch.Tensor: Physical Potential Vorticity * f0.
                └── (n_ens, nl, nx-1, ny-1)-shaped.
        """
        # Compute ω = ∂_x v - ∂_y u
        omega = torch.diff(uvh.v[..., 1:-1], dim=-2) - torch.diff(
            uvh.u[..., 1:-1, :],
            dim=-1,
        )
        # Compute ω-f_0*h/H
        return (omega - f0 * points_to_surfaces(uvh.h) / H[0, 0, 0]) * (
            f0 / ds
        )

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
            f0=self._f0,
            H=self.H,
            ds=self._space.ds,
            points_to_surfaces=self._points_to_surface,
        )

    def QoG_inv(  # noqa: N802
        self,
        elliptic_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inversion of Q∘G.

        Args:
            elliptic_rhs (torch.Tensor): Right hand side,
                └── (n_ens, nl, nx-1, ny-1)-shaped.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pressure p and
            interpolated pressure p_i.
                ├── p: (n_ens, nl, nx+1, ny+1)-shaped
                └── p_i: (n_ens, nl, nx, ny)-shaped
        """
        pi_i = torch.zeros(
            (1, 1, self._space.nx, self._space.ny),
            dtype=elliptic_rhs.dtype,
            device=elliptic_rhs.device,
        )
        A_12 = compute_A_12(self.H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806

        verbose.display(
            f"[{self.__class__.__name__}.QoG_inv]: "
            "Retrieving pressure using iterative solving "
            f"with at most {self._MAX_ITERATIONS} iterations.",
            trigger_level=3,
        )

        for k in range(1, self._MAX_ITERATIONS + 1):
            # Since A.shape = (1,1) -> solving in mode space is the same
            # as solving in physical space
            pi1 = self._compute_p_modes(
                elliptic_rhs
                + (
                    self._f0**2
                    * A_12
                    * self._points_to_surface(pi_i)
                    * self._points_to_surface(self.alpha)
                ),
            )

            if torch.isnan(pi1).any():
                msg = f"Overflow after {k} iterations."
                raise OverflowError(msg)

            pi1_i = self._points_to_surface(pi1)

            if torch.isclose(
                pi1_i,
                pi_i,
                atol=self._ATOL,
                rtol=self._RTOL,
            ).all():
                verbose.display(
                    f"[{self.__class__.__name__}.QoG_inv]: "
                    f"Convergence reached after {k} iterations.",
                    trigger_level=3,
                )
                break
            pi_i = pi1_i

            if k == self._MAX_ITERATIONS:
                msg = "Max iterations reached, no convergence."
                raise RuntimeError(msg)

        # Add homogeneous solutions to ensure mass conservation
        gamma = -pi1.mean((-1, -2), keepdim=True) / self.homsol_wgrid_mean
        pi1 += gamma * self.homsol_wgrid

        p_qg_i = self._points_to_surface(pi1)
        return pi1, p_qg_i

    def to_shape(self, nx: int, ny: int) -> Self:
        """Recreate a QGProjector with another shape.

        Args:
            nx (int): New nx.
            ny (int): New ny.

        Returns:
            Self: QGProjector.
        """
        proj = CollinearSFProjector(
            A=self.A,
            H=self.H,
            g_prime=self._g_prime,
            space=self.space.to_shape(nx, ny, self.space.nl),
            f0=self._f0,
            masks=self.masks,
        )
        alpha = self.alpha
        proj.alpha = interpolate_physical_variable(alpha, (nx, ny))
        return proj

    @classmethod
    def from_config(
        cls,
        space_config: SpaceConfig,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Self:
        """Builds Projector frm configuration.

        Args:
            space_config (SpaceConfig): Space configuration.
            model_config (ModelConfig): Model configuration.
            physics_config (PhysicsConfig): _description_
            dtype (torch.dtype | None, optional): Dtype. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.

        Returns:
            Self: CollinearSFProjector.
        """
        specs = defaults.get(dtype=dtype, device=device)
        g_tilde = compute_g_tilde(model_config.g_prime)
        return cls(
            compute_A(model_config.h[:1], g_tilde, **specs),
            model_config.h.unsqueeze(-1).unsqueeze(-1),
            model_config.g_prime.unsqueeze(-1).unsqueeze(-1),
            SpaceDiscretization3D.from_config(space_config, model_config),
            physics_config.f0,
            masks=Masks.empty(
                space_config.nx,
                space_config.ny,
                device=specs["device"],
            ),
        )
