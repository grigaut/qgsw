"""Modified QG Model with Collinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.base import Model
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.qg.alpha import Coefficient, ConstantCoefficient
from qgsw.models.qg.core import QG
from qgsw.models.sw import SW
from qgsw.models.variables import UVH
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.specs import DEVICE
from qgsw.utils.gaussian_filtering import GaussianFilter2D

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane


class _QGCollinearSublayer(QG):
    """Collinear QG Model."""

    _supported_layers_nb: int
    _coefficient: float = 0.01

    def __init__(
        self,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        coefficient: float | Coefficient = _coefficient,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Collinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Collinearity coefficient.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        Model.__init__(
            self,
            space_3d=space_3d,
            g_prime=g_prime,
            beta_plane=beta_plane,
            optimize=optimize,
        )
        self._core = self._init_core_model(optimize=optimize)
        self.coefficient = coefficient

    @property
    def coefficient(self) -> Coefficient:
        """Collinearity coefficient."""
        return self._coefficient

    @coefficient.setter
    def coefficient(self, coefficient: float | Coefficient) -> None:
        return self._set_coefficient(coefficient)

    @property
    def alpha(self) -> float:
        """Alpha value."""
        return self.coefficient.at_current_time()

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H[0, ...].unsqueeze(0)

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime[0, ...].unsqueeze(0)

    def _set_H(self, h: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Perform additional validation over H.

        Args:
            h (torch.Tensor): Layers thickness.

        Raises:
            ValueError: if H is not constant in space

        Returns:
            torch.Tensor: H
        """
        if self.space.nl != self._supported_layers_nb:
            msg = (
                "_QGCollinearSublayer can only support"
                f"{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        super()._set_H(h)

    def _set_coefficient(self, coefficient: float | Coefficient) -> None:
        """Set collinearity coefficient value.

        Args:
            coefficient (float): Collinearity Coefficient.
        """
        if isinstance(coefficient, int | float):
            self._coefficient = ConstantCoefficient(coefficient)
        else:
            self._coefficient = coefficient

    def _init_core_model(self, optimize: bool) -> SW:  # noqa: FBT001
        """Initialize the core Shallow Water model.

        Args:
            optimize (bool): Wehether to optimize the model functions or not.

        Returns:
            SW: Core model.
        """
        return SW(
            space_3d=keep_top_layer(self._space),
            g_prime=self.g_prime,
            beta_plane=self.beta_plane,
            optimize=optimize,
        )


class QGCollinearSF(_QGCollinearSublayer):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _supported_layers_nb: int = 2
    _coefficient = 0.01

    def __init__(
        self,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        coefficient: float | Coefficient = _coefficient,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Collinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Collinearity coefficient.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(
            space_3d=space_3d,
            g_prime=g_prime,
            beta_plane=beta_plane,
            coefficient=coefficient,
            optimize=optimize,
        )

    def _set_coefficient(self, coefficient: float | Coefficient) -> None:
        super()._set_coefficient(coefficient)
        self._update_A()

    def _update_A(self) -> None:  # noqa: N802
        """Update the stretching operator matrix."""
        self.A = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])
        decomposition = self.compute_layers_to_mode_decomposition(self.A)
        self.Cm2l, self.lambd, self.Cl2m = decomposition
        self.set_helmholtz_solver(self.lambd)

    def compute_A(  # noqa: N802
        self,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
    ) -> torch.Tensor:
        """Compute new Stretching operator.

        Ã = (1/ρ_1)[[(1/H_1)*(1/g_1 + (1 - α)/g_2)]]

        Args:
            H (torch.Tensor): Layers reference height.
            g_prime (torch.Tensor): Reduced gravity values.

        Returns:
            torch.Tensor: Stretching Operator
        """  # noqa: RUF002
        A = super().compute_A(H, g_prime)  # noqa: N806
        # Create layers coefficients vector [1, α]  # noqa: RUF003
        layers_coefs = torch.tensor(
            [1, self.alpha],
            dtype=self.dtype,
            device=self.device.get(),
        )
        # Select top row from matrix product
        return (A @ layers_coefs)[0, ...].unsqueeze(0).unsqueeze(0)

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        if not self.coefficient.isconstant:
            self.coefficient.next_time(self.dt)
            self._update_A()
        return super().step()


class QGCollinearPV(_QGCollinearSublayer):
    """Modified QG Model implementing collinear pv behavior."""

    _supported_layers_nb: int = 2
    _coefficient = 0.01

    def __init__(
        self,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        coefficient: float | Coefficient = _coefficient,
        optimize: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """Collinear Sublayer Potential Vorticity.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Collinearity coefficient.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(space_3d, g_prime, beta_plane, coefficient, optimize)
        # Two layers stretching operator for QoG inversion
        A_2l = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806
        decomposition = self.compute_layers_to_mode_decomposition(A_2l)
        self.Cm2l, self.lambd, self.Cl2m = decomposition
        # Two layers helmholtz solver
        self.set_helmholtz_solver(self.lambd)
        # One layer stretching operator for G
        self.A = self.compute_A(self.H[:, 0, 0], self.g_prime[:, 0, 0])

    def QoG_inv(  # noqa: N802
        self,
        elliptic_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(Q o G)^{-1} operator: solve elliptic eq with mass conservation.

        Modified implementation: Expand to 2 layers and retrieve
        top layer streamfunction.

        Args:
            elliptic_rhs (torch.Tensor): Elliptic equation right hand side
            value (ω-f_0*h/H).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Quasi-geostrophique pressure,
            interpolated quasi-geostroophic pressure ("middle of grid cell").
        """
        # Expand to 2 layers
        _, _, nx, ny = elliptic_rhs.shape
        elliptic_rhs_2l = torch.zeros(
            (1, 2, nx, ny),
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        elliptic_rhs_2l[0, 0, ...] = elliptic_rhs
        elliptic_rhs_2l[0, 1, ...] = self.alpha * elliptic_rhs
        # Extract 2 layers stream functions
        p_qg_2l, p_qg_i_2l = super().QoG_inv(elliptic_rhs_2l)
        # Shrink to 1 layer
        return p_qg_2l[:, 0, ...], p_qg_i_2l[:, 0, ...]

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        if not self.coefficient.isconstant:
            self.coefficient.next_time(self.dt)
        return super().step()


class QGSmoothCollinearSF(_QGCollinearSublayer):
    """QGSmoothCollinearSF."""

    _coefficient = 0
    _supported_layers_nb = 2

    def __init__(
        self,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
        coefficient: float | Coefficient = _coefficient,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Smoothed Collinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Collinearity coefficient.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(space_3d, g_prime, beta_plane, coefficient, optimize)
        # Two layers stretching operator for QoG inversion
        A_2l = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806
        decomposition = self.compute_layers_to_mode_decomposition(A_2l)
        self.Cm2l, self.lambd, self.Cl2m = decomposition
        # Two layers helmholtz solver
        self.set_helmholtz_solver(self.lambd)
        # One layer stretching operator for G
        self.A = self.compute_A(self.H[:, 0, 0], self.g_prime[:, 0, 0])
        self._gaussian_filter = GaussianFilter2D(3, 10)

    def G(self, p: torch.Tensor, p_i: torch.Tensor | None = None) -> UVH:  # noqa: N802
        """G operator.

        Args:
            p (torch.Tensor): Pressure.
            p_i (Union[None, torch.Tensor], optional): Interpolated pressure
             ("middle of grid cell"). Defaults to None.

        Returns:
            UVH: u, v and h
        """
        uvh = super().G(p=p, p_i=p_i)
        return UVH(uvh.u[:, [0], ...], uvh.v[:, [0], ...], uvh.h[:, [0], ...])

    def Q(self, uvh: UVH) -> torch.Tensor:  # noqa: N802
        """Q operator: compute elliptic equation r.h.s.

        Args:
            uvh (UVH): u,v and h.

        Returns:
            torch.Tensor: Elliptic equation right hand side (ω-f_0*h/H).
        """
        alpha = self.alpha
        smooth_u = self._gaussian_filter.smooth(uvh.u[0, 0, ...])
        u_2l = torch.cat(
            [uvh.u[0, [0], ...], alpha * smooth_u.unsqueeze(0)],
            dim=0,
        ).unsqueeze(0)
        smooth_v = self._gaussian_filter.smooth(uvh.v[0, 0, ...])
        v_2l = torch.cat(
            [uvh.v[0, [0], ...], alpha * smooth_v.unsqueeze(0)],
            dim=0,
        ).unsqueeze(0)
        smooth_h = self._gaussian_filter.smooth(uvh.h[0, 0, ...])
        h_2l = torch.cat(
            [uvh.h[0, [0], ...], alpha * smooth_h.unsqueeze(0)],
            dim=0,
        ).unsqueeze(0)

        return super().Q(UVH(u_2l, v_2l, h_2l))
