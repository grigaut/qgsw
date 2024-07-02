"""Modified QG Model with Colinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.base import Model
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.qg.alpha import Coefficient, ConstantCoefficient
from qgsw.models.qg.core import QG
from qgsw.models.sw import SW
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
    keep_top_layer,
)
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane


class _QGColinearSublayer(QG):
    """Colinear QG Model."""

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
        """Colinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Colinearity coefficient.
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
        """Colinearity coefficient."""
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
                "_QGColinearSublayer can only support"
                f"{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        super()._set_H(h)

    def _set_coefficient(self, coefficient: float | Coefficient) -> None:
        """Set colinearity coefficient value.

        Args:
            coefficient (float): Colinearity Coefficient.
        """
        if isinstance(coefficient, (int, float)):
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


class QGColinearSublayerStreamFunction(_QGColinearSublayer):
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
        """Colinear Sublayer Stream Function.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Colinearity coefficient.
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
        if self.coefficient.isconstant:
            self._update_A()

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
            device=self.device,
        )
        # Select top row from matrix product
        return (A @ layers_coefs)[0, ...].unsqueeze(0).unsqueeze(0)

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        if not self.coefficient.isconstant:
            self.coefficient.next_time(self.dt)
            self._update_A()
        return super().step()


class QGColinearSublayerPV(_QGColinearSublayer):
    """Modified QG Model implementing potential vorticity colinear behavior."""

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
        """Colinear Sublayer Potential Vorticity.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
            coefficient (float): Colinearity coefficient.
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
            device=DEVICE,
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


class QGPVMixture(QGColinearSublayerPV):
    """Mixture of Barotropic and Baroclinic Streamfunctions."""

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
        elliptic_rhs_baroclinic = torch.zeros(
            (1, 2, nx, ny),
            device=DEVICE,
            dtype=torch.float64,
        )
        elliptic_rhs_baroclinic[0, 0, ...] = elliptic_rhs
        elliptic_rhs_baroclinic[0, 1, ...] = 0 * elliptic_rhs
        # Extract 2 layers stream functions
        p_qg_baroclinic, p_qg_i_baroclinic = _QGColinearSublayer.QoG_inv(
            self,
            elliptic_rhs_baroclinic,
        )

        elliptic_rhs_barotropic = torch.zeros(
            (1, 2, nx, ny),
            device=DEVICE,
            dtype=torch.float64,
        )
        elliptic_rhs_barotropic[0, 0, ...] = elliptic_rhs
        elliptic_rhs_barotropic[0, 1, ...] = 1 * elliptic_rhs
        # Extract 2 layers stream functions
        p_qg_barotropic, p_qg_i_barotropic = _QGColinearSublayer.QoG_inv(
            self,
            elliptic_rhs_barotropic,
        )

        # Shrink to 1 layer
        p_qg_barocl_top = p_qg_baroclinic[:, 0, ...]
        p_qg_barotr_top = p_qg_barotropic[:, 0, ...]
        p_qg_i_barocl_top = p_qg_i_baroclinic[:, 0, ...]
        p_qg_i_barotr_top = p_qg_i_barotropic[:, 0, ...]

        alpha = self.alpha

        p_qg = alpha * p_qg_barotr_top + (1 - alpha) * p_qg_barocl_top
        p_qg_i = alpha * p_qg_i_barotr_top + (1 - alpha) * p_qg_i_barocl_top

        return p_qg, p_qg_i
