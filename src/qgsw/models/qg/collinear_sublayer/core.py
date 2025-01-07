"""Modified QG Model with Collinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import UVH
from qgsw.models.base import Model
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.io import IO
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.collinear_sublayer.alpha import (
    Coefficient,
    ConstantCoefficient,
)
from qgsw.models.qg.collinear_sublayer.stretching_matrix import (
    compute_A_collinear_sf,
)
from qgsw.models.qg.core import QG
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.models.sw.core import SW
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)
from qgsw.specs import DEVICE
from qgsw.utils.gaussian_filtering import GaussianFilter2D

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )


class _QGCollinearSublayer(QG):
    """Collinear QG Model."""

    _supported_layers_nb: int
    _coefficient: float
    _A: torch.Tensor

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Collinear Sublayer Stream Function.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        ModelParamChecker.__init__(
            self,
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
        )
        self._space = keep_top_layer(self._space)
        ##Topography and Ref values
        self._set_ref_variables()

        # initialize state
        self._state = State.steady(
            n_ens=self.n_ens,
            nl=self.space.nl,
            nx=self.space.nx,
            ny=self.space.ny,
            dtype=self.dtype,
            device=self.device.get(),
        )
        self._io = IO(u=self._state.u, v=self._state.v, h=self._state.h)
        # initialize variables
        self._create_diagnostic_vars(self._state)

        self._set_utils(optimize)
        self._set_fluxes(optimize)
        self._core = self._init_core_model(
            space_2d=space_2d,
            H=H,
            g_prime=g_prime,
            optimize=optimize,
        )

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
        return self._H[:1, ...]

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime[:1, ...]

    def _init_core_model(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool,  # noqa: FBT001
    ) -> SW:
        """Initialize the core Shallow Water model.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.

        Returns:
            SW: Core model (One layer).
        """
        return SW(
            space_2d=space_2d,
            H=H[:1],  # Only consider top layer
            g_prime=g_prime[:1],  # Only consider top layer
            optimize=optimize,
        )

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
        if isinstance(coefficient, (int, float)):
            self._coefficient = ConstantCoefficient(coefficient)
        else:
            self._coefficient = coefficient


class QGCollinearSF(_QGCollinearSublayer):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _supported_layers_nb: int = 2
    _beta_plane_set = False
    _coefficient_set = False

    @Model.beta_plane.setter
    def beta_plane(self, beta_plane: BetaPlane) -> None:
        """Beta-plane setter."""
        Model.beta_plane.fset(self, beta_plane)
        self.sw.beta_plane = beta_plane
        self._beta_plane_set = True
        if self._coefficient_set:
            self.set_helmholtz_solver(self.lambd)
            self._create_diagnostic_vars(self._state)

    @_QGCollinearSublayer.coefficient.setter
    def coefficient(self, coefficient: BetaPlane) -> None:
        """Beta-plane setter."""
        _QGCollinearSublayer.coefficient.fset(self, coefficient)

    def _set_coefficient(self, coefficient: float | Coefficient) -> None:
        super()._set_coefficient(coefficient)
        self._coefficient_set = True
        self._update_A()

    def _update_A(self) -> None:  # noqa: N802
        """Update the stretching operator matrix."""
        self.A = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])
        decomposition = compute_layers_to_mode_decomposition(self.A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
        if self._beta_plane_set:
            self.set_helmholtz_solver(self.lambd)
            self._create_diagnostic_vars(self._state)

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
        return compute_A_collinear_sf(
            H=H,
            g_prime=g_prime,
            alpha=self.alpha,
            dtype=self.dtype,
            device=self.device.get(),
        )

    def step(self) -> None:
        """Performs one step time-integration with RK3-SSP scheme."""
        if not self.coefficient.isconstant:
            self.coefficient.next_time(self.dt)
            self._update_A()
        return super().step()


class QGCollinearPV(_QGCollinearSublayer):
    """Modified QG Model implementing collinear pv behavior."""

    _supported_layers_nb: int = 2

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,  # noqa: FBT002, FBT001
    ) -> None:
        """Collinear Sublayer Potential Vorticity.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(space_2d, H, g_prime, optimize)
        # Two layers stretching operator for QoG inversion
        A_2l = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806
        decomposition = compute_layers_to_mode_decomposition(A_2l)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
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

    _supported_layers_nb = 2

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        optimize: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Smoothed Collinear Sublayer Stream Function.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
            optimize (bool, optional): Whether to precompile functions or
            not. Defaults to True.
        """
        super().__init__(space_2d, H, g_prime, optimize)
        # Two layers stretching operator for QoG inversion
        A_2l = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])  # noqa: N806
        decomposition = compute_layers_to_mode_decomposition(A_2l)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
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
