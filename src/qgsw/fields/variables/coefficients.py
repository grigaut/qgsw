"""Collinearity Coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    Pressure,
)
from qgsw.models.qg.core import QG
from qgsw.models.sw.core import SW

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from abc import ABC, abstractmethod

import torch

from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.fields.variables.dynamics import StreamFunction
from qgsw.specs import DEVICE
from qgsw.utils.least_squares_regression import (
    perform_linear_least_squares_regression,
)
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.configs.core import Configuration
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.state import State
    from qgsw.fields.variables.uvh import BasePrognosticTuple
    from qgsw.models.qg.modified.collinear_sublayer.core import QGAlpha


class Coefficient(DiagnosticVariable, ABC):
    """Coefficient."""

    _unit = Unit._
    _scope = Scope.ENSEMBLE_WISE

    def update_model(self, model: QGAlpha) -> None:
        """Update a model coefficient value.

        Args:
            model (QGAlpha): Model to update.
        """
        if model.get_type() == QG.get_type():
            return
        if model.get_type() == SW.get_type():
            return
        model.alpha = self.compute_no_slice(model.prognostic)

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,
        space_config: SpaceConfig,
    ) -> Self:
        """Instantiate coeffcient from configuration.

        Args:
            model_config (ModelConfig): Model configuration.
            physics_config (PhysicsConfig): Physics configuration.
            space_config (SpaceConfig): Space configuration.

        Returns:
            Self: Coefficient.
        """


class LSRSFInferredAlpha(Coefficient):
    """Inferred collinearity from the streamfunction.

    Performs linear least squares regression to infer alpha.
    """

    _name = "alpha_lsr_sf"
    _description = "LSR-Stream function inferred coefficient"

    def __init__(self, psi_ref: StreamFunction) -> None:
        """Instantiate the variable.

        Args:
            psi_ref (StreamFunction): Reference stream function.
        """
        self._psi = psi_ref

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of alpha.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            Tensor: Alpha
        """
        psi = self._psi.compute_no_slice(prognostic)
        psi_1 = psi[:, 0, ...]  # (n_ens,nx,ny)-shaped
        psi_2 = psi[:, 1, ...]  # (n_ens,nx,ny)-shaped

        x = psi_1.flatten(-2, -1).unsqueeze(-1)  # (n_ens,nx*ny,1)-shaped
        y = psi_2.flatten(-2, -1)  # (n_ens,nx*ny)-shaped

        try:
            return perform_linear_least_squares_regression(x, y)[:, 0]
        except torch.linalg.LinAlgError:
            return torch.zeros(
                (y.shape[0],),
                dtype=torch.float64,
                device=DEVICE.get(),
            )

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,
        space_config: SpaceConfig,
    ) -> Self:
        """Instantiate coeffcient from configuration.

        Args:
            model_config (ModelConfig): Model configuration.
            physics_config (PhysicsConfig): Physics configuration.
            space_config (SpaceConfig): Space configuration.

        Returns:
            Self: Coefficient.
        """
        h_phys = PhysicalLayerDepthAnomaly(space_config.ds)
        eta_phys = PhysicalSurfaceHeightAnomaly(h_phys)
        p = Pressure(
            g_prime=model_config.g_prime.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1),
            eta_phys=eta_phys,
        )
        psi = StreamFunction(p, physics_config.f0)
        return cls(psi)


class ConstantCoefficient(Coefficient):
    """Constant collinearity coefficient."""

    _name = "alpha_constant"
    _description = "Constant coefficient"

    def __init__(self, value: torch.Tensor) -> None:
        """Instantiate the coefficient.

        Args:
            value (torch.Tensor): Coefficient value.
        """
        self._value = value

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:  # noqa: ARG002
        """Compute the value of alpha.

        Args:
            prognostic (BasePrognosticTuple): Useless, for compatibility
            reasons.

        Returns:
            Tensor: Alpha.
        """
        return self._value

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        physics_config: PhysicsConfig,  # noqa: ARG003
        space_config: SpaceConfig,  # noqa: ARG003
    ) -> Self:
        """Instantiate coeffcient from configuration.

        Args:
            model_config (ModelConfig): Model configuration.
            physics_config (PhysicsConfig): Physics configuration,
            for compatibility.
            space_config (SpaceConfig): Space configuration,
            for compatibility.

        Returns:
            Self: Coefficient.
        """
        return cls(
            torch.tensor(
                [model_config.collinearity_coef.value],
                dtype=torch.float64,
                device=DEVICE.get(),
            ),
        )


def create_coefficient(
    config: Configuration,
) -> ConstantCoefficient | LSRSFInferredAlpha:
    """Create the coefficient.

    Args:
        config (Configuration): Model Configuration.

    Raises:
        ValueError: If the coefficient is not valid.

    Returns:
        ConstantCoefficient | LSRSFInferredAlpha: Coefficient
    """
    coef_type = config.model.collinearity_coef.type
    if coef_type == ConstantCoefficient.get_name():
        return ConstantCoefficient.from_config(
            model_config=config.model,
            physics_config=config.physics,
            space_config=config.space,
        )
    if coef_type == LSRSFInferredAlpha.get_name():
        return LSRSFInferredAlpha.from_config(
            model_config=config.simulation.reference,
            physics_config=config.physics,
            space_config=config.space,
        )
    msg = "Possible coeffciient types are: "
    coef_types = [
        ConstantCoefficient.get_name(),
        LSRSFInferredAlpha.get_name(),
    ]
    raise ValueError(msg + ", ".join(coef_types))
