"""Modified QG model with filtered top layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.exceptions import InvalidLayerNumberError
from qgsw.logging import getLogger
from qgsw.models.names import ModelName
from qgsw.models.parameters import ModelParamChecker
from qgsw.models.qg.uvh.modified.collinear.core import QGAlpha
from qgsw.models.qg.uvh.modified.filtered.pv import (
    compute_g_tilde,
)
from qgsw.models.qg.uvh.modified.filtered.variable_set import (
    QGCollinearFilteredSFVariableSet,
)
from qgsw.models.qg.uvh.projectors.filtered import (
    CollinearFilteredPVProjector,
    CollinearFilteredSFProjector,
)
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    keep_top_layer,
)
from qgsw.utils.shape_checks import with_shapes

if TYPE_CHECKING:
    import torch

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.physics.coriolis.beta_plane import BetaPlane


logger = getLogger(__name__)


class QGCollinearFilteredSF(QGAlpha[CollinearFilteredSFProjector]):
    """Modified QG Model implementing collinear sf behavior."""

    _type = ModelName.QG_FILTERED_SF
    _supported_layers_nb = 2

    @with_shapes(H=(2,), g_prime=(2,))
    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,
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
        msg = f"Creating {self.__class__.__name__} model..."
        logger.info(msg)
        self.__instance_nb = next(self._instance_count)
        self.name = f"{self.__class__.__name__}-{self.__instance_nb}"
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
    def alpha(self, alpha: torch.Tensor) -> None:
        """Setter for alpha."""
        QGAlpha.alpha.fset(self, alpha)
        self._P.alpha = alpha
        self._create_diagnostic_vars(self._state)

    def set_p(
        self,
        p: torch.Tensor,
        offset_p0: torch.Tensor | None = None,
        offset_p1: torch.Tensor | None = None,
    ) -> None:
        """Set the initial pressure.

        The pressure must contain at least as many layers as the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl, nx+1, ny+1)-shaped
            offset_p0 (torch.Tensor): Offset for the pressure in top layer.
                └── (1, 1, nx, ny)-shaped
            offset_p1 (torch.Tensor): Offset for the pressure in bottom layer.
                └── (1, 1, nx, ny)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        uvh = self.P.G(
            p[:, :nl],
            self.A,
            self.H,
            self._g_prime,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.alpha,
            self.P.filter,
            self.interpolate,
            offset_p0=offset_p0 if offset_p0 is not None else self.P.offset_p0,
            offset_p1=offset_p1 if offset_p0 is not None else self.P.offset_p1,
        )
        self.set_uvh(*uvh)

    def _set_projector(self) -> None:
        self._P = CollinearFilteredSFProjector(
            A=self.A,
            H=self.H,
            g_prime=self.g_prime,
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
        return QGCollinearFilteredSFVariableSet.get_variable_set(
            space,
            physics,
            model,
        )


class QGCollinearFilteredPV(QGAlpha[CollinearFilteredPVProjector]):
    """Modified QG Model implementing collinear pv behavior."""

    _type = ModelName.QG_FILTERED_PV
    _supported_layers_nb = 2

    @with_shapes(H=(2,), g_prime=(2,))
    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,
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
        msg = f"Creating {self.__class__.__name__} model..."
        logger.info(msg)
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
            H=H[:1],
            g_prime=g_prime[:1],
            beta_plane=beta_plane,
            optimize=optimize,
        )
        self.A = self.compute_A(
            H,
            g_prime,
        )
        self._set_projector()

    @QGAlpha.alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        """Setter for alpha."""
        QGAlpha.alpha.fset(self, alpha)
        self._P.alpha = alpha
        self._create_diagnostic_vars(self._state)

    def set_p(
        self,
        p: torch.Tensor,
    ) -> None:
        """Set the initial pressure.

        The pressure must contain at least more than 1 layer than the model.

        Args:
            p (torch.Tensor): Pressure.
                └── (n_ens, >= nl +1 , nx+1, ny+1)-shaped

        Raises:
            InvalidLayerNumberError: If the layer number of p is invalid.
        """
        if p.shape[1] < (nl := self.space.nl + 1):
            msg = f"p must have at least {nl} layers."
            raise InvalidLayerNumberError(msg)

        uvh = self.P.G(
            p[:, :nl],
            self.A,
            self.H,
            self._space.dx,
            self._space.dy,
            self._space.ds,
            self.beta_plane.f0,
            self.interpolate,
        )
        self.set_uvh(*uvh)

    def _set_projector(self) -> None:
        self._P = CollinearFilteredPVProjector(
            A=self.A,
            H=self.H,
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
