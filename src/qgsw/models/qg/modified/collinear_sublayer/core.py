"""Modified QG Model with Collinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw import verbose
from qgsw.fields.variables.state import StateAlpha
from qgsw.fields.variables.uvh import UVHTAlpha
from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.io import IO
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.core import QGCore
from qgsw.models.qg.modified.collinear_sublayer.stretching_matrix import (
    compute_A_collinear_sf,
)
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.models.sw.core import SWCollinearSublayer
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)

if TYPE_CHECKING:
    import torch

    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )


class QGAlpha(QGCore[UVHTAlpha]):
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
        self._state = StateAlpha.steady(
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
                f"QGAlpha can only support{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        super()._set_H(h)


class QGCollinearSF(QGAlpha):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _type = "QGCollinearSF"

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
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
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

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H[:1, ...]

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime[:1, ...]

    @QGAlpha.alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        """Beta-plane setter."""
        QGAlpha.alpha.fset(self, alpha)
        self._core.alpha = alpha
        self._update_A()

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
            H (torch.Tensor): Reference layer depths tensor, (nl,) shaped.
            g_prime (torch.Tensor): Reduced Gravity Tensor, (nl,) shaped.
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

    def _update_A(self) -> None:  # noqa: N802
        """Update the stretching operator matrix."""
        self.A = self.compute_A(self._H[:, 0, 0], self._g_prime[:, 0, 0])
        decomposition = compute_layers_to_mode_decomposition(self.A)
        self.Cm2l, lambd, self.Cl2m = decomposition
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
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
        return super().step()
