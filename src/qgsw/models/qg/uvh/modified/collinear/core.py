"""Modified QG Model with Collinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from torch.utils import checkpoint

from qgsw import verbose
from qgsw.exceptions import (
    InvalidLayerNumberError,
    InvalidLayersDefinitionError,
)
from qgsw.fields.variables.state import StateUVHAlpha
from qgsw.fields.variables.tuples import UVH, UVHTAlpha
from qgsw.models.core import time_steppers
from qgsw.models.io import IO
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.stretching_matrix import (
    compute_A,
)
from qgsw.models.qg.uvh.core import QGCore
from qgsw.models.qg.uvh.modified.collinear.variable_set import (
    QGCollinearSFVariableSet,
)
from qgsw.models.qg.uvh.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.uvh.projectors.collinear import (
    CollinearPVProjector,
    CollinearSFProjector,
)
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sw.core import SWCollinearSublayer
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
    )

Projector = TypeVar("Projector", bound=QGProjector)


class QGAlpha(QGCore[UVHTAlpha, StateUVHAlpha, Projector]):
    """Collinear QG Model."""

    _supported_layers_nb: int
    _A: torch.Tensor
    _requires_grad = False

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        return self._state.alpha.get()

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._state.update_alpha(alpha)
        self.requires_grad_(self.alpha.requires_grad)

    @property
    def requires_grad(self) -> bool:
        """Is True if gradients need to be computed, False otherwise."""
        return self._requires_grad

    def _set_io(self, state: StateUVHAlpha) -> None:
        self._io = IO(
            state.t,
            state.alpha,
            *state.physical,
        )

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

    def update(self, prognostic: UVH) -> UVH:
        """Update prognostic.

        Args:
            prognostic (UVH): u,v and h.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            UVH: update prognostic variables.
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        if self.save_p_values:
            self._p_vals = []

        def time_integration(
            u: torch.Tensor,
            v: torch.Tensor,
            h: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return time_steppers.rk3_ssp(
                UVH(u, v, h),
                self.dt,
                self.compute_time_derivatives,
            )

        if self.requires_grad:
            u, v, h = prognostic
            u_grad = u.clone().requires_grad_(True)  # noqa: FBT003
            v_grad = v.clone().requires_grad_(True)  # noqa: FBT003
            h_grad = h.clone().requires_grad_(True)  # noqa: FBT003
            uvh_tuple = checkpoint.checkpoint(
                time_integration,
                u_grad,
                v_grad,
                h_grad,
                use_reentrant=True,
            )
            return UVH(*uvh_tuple)
        return time_integration(prognostic.u, prognostic.v, prognostic.h)

    def requires_grad_(self, requires_grad: bool) -> None:  # noqa: FBT001
        """Set requires_grad attribute to True.

        Hence, the model will compute checkpoints when doing update.

        Args:
            requires_grad (bool): True if gradients need to be computed,
                False otherwise.
        """
        self._requires_grad = requires_grad
        if self.requires_grad:
            verbose.display(
                msg=(
                    "Model set to track gradients."
                    " Model steps will register pytorch checkpoints."
                ),
                trigger_level=1,
            )


class QGCollinearSF(QGAlpha[CollinearSFProjector]):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _type = ModelName.QG_COLLINEAR_SF

    _supported_layers_nb: int = 2
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
        self.__instance_nb = next(self._instance_count)
        self.name = f"{self.__class__.__name__}-{self.__instance_nb}"
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

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        uvh = self.P.G(
            p[:, :nl],
            self.A,
            self._H,
            self._g_prime,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.alpha,
            self.interpolate,
        )
        self.set_uvh(*uvh)

    def _set_projector(self) -> None:
        self._P = CollinearSFProjector(
            self.A,
            self._H,
            g_prime=self._g_prime,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

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


class QGCollinearPV(QGAlpha[CollinearPVProjector]):
    """QG model with collinear potential vorticity."""

    _type = ModelName.QG_COLLINEAR_PV

    _supported_layers_nb: int = 2
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
        self.__instance_nb = next(self._instance_count)
        self.name = f"{self.__class__.__name__}-{self.__instance_nb}"
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

    def set_p(self, p: torch.Tensor) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        uvh = self.P.G(
            p[:, :nl],
            self.A,
            self._H,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.interpolate,
        )
        self.set_uvh(*uvh)

    def _set_projector(self) -> None:
        self._P = CollinearPVProjector(
            self.A,
            self._H,
            space=self.space,
            f0=self.beta_plane.f0,
            masks=self.masks,
        )

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
        raise NotImplementedError
