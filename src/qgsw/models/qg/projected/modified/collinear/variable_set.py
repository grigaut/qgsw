"""Variable set related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    CollinearityCoefficientDiag,
    PhysicalSurfaceHeightAnomaly,
    PhysicalVorticity,
    PotentialVorticity,
    PressureTilde,
    Vorticity,
)
from qgsw.models.qg.projected.variable_set import QGVariableSet

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGCollinearSFVariableSet(QGVariableSet):
    """QgCollinearSF Variable Set."""

    @classmethod
    def add_prognostics(cls, var_dict: dict[str, DiagnosticVariable]) -> None:
        """Add prognostic variables.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dict.
        """
        super().add_prognostics(var_dict)
        var_dict[CollinearityCoefficientDiag.get_name()] = (
            CollinearityCoefficientDiag()
        )

    @classmethod
    def add_pressure(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        model: ModelConfig,
    ) -> None:
        """Add pressure.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
            model (ModelConfig): Model Configuration.
        """
        super().add_pressure(var_dict, model)

        var_dict[PressureTilde.get_name()] = PressureTilde(
            model.g_prime.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            var_dict[PhysicalSurfaceHeightAnomaly.get_name()],
        )

    @classmethod
    def add_vorticity(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
        model: ModelConfig,
        physics: PhysicsConfig,
    ) -> None:
        """Add vorticity.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
            space (SpaceConfig): Space configuration.
            model (ModelConfig): Model Configuration.
            physics (PhysicsConfig): Physics Confdiguration.
        """
        var_dict[Vorticity.get_name()] = Vorticity()
        var_dict[PhysicalVorticity.get_name()] = PhysicalVorticity(
            var_dict[Vorticity.get_name()],
            space.ds,
        )
        var_dict[PotentialVorticity.get_name()] = PotentialVorticity(
            model.h[:1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            space.ds,
            physics.f0,
        )

    @classmethod
    def add_energy(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
        model: ModelConfig,
        physics: PhysicsConfig,
    ) -> None:
        """Add energy.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
            space (SpaceConfig): Space configuration.
            model (ModelConfig): Model Configuration.
            physics (PhysicsConfig): Physics Configuration.
        """
        # ruff: noqa: ERA001
        # if model.collinearity_coef.type == CoefficientName.UNIFORM or (
        #     model.collinearity_coef.type
        #     == CoefficientName.LSR_INFERRED_UNIFORM
        # ):
        #     alpha = model.collinearity_coef.initial
        #
        # A = compute_A_collinear_sf(
        #     model.h,
        #     model.g_prime,
        #     alpha,
        #     torch.float64,
        #     DEVICE.get(),
        # )
        # var_dict[ModalKineticEnergy.get_name()] = ModalKineticEnergy(
        #     A,
        #     var_dict[StreamFunction.get_name()],
        #     model.h[:1],
        #     space.dx,
        #     space.dy,
        # )
        # var_dict[ModalAvailablePotentialEnergy.get_name()] = (
        #     ModalAvailablePotentialEnergy(
        #         A,
        #         var_dict[StreamFunction.get_name()],
        #         model.h[:1],
        #         physics.f0,
        #     )
        # )
        # var_dict[ModalEnergy.get_name()] = ModalEnergy(
        #     var_dict[ModalKineticEnergy.get_name()],
        #     var_dict[ModalAvailablePotentialEnergy.get_name()],
        # )
        # var_dict[TotalKineticEnergy.get_name()] = TotalKineticEnergy(
        #     var_dict[StreamFunction.get_name()],
        #     model.h[:1],
        #     space.dx,
        #     space.dy,
        # )
        # var_dict[TotalAvailablePotentialEnergy.get_name()] = (
        #     TotalAvailablePotentialEnergy(
        #         A,
        #         var_dict[StreamFunction.get_name()],
        #         model.h[:1],
        #         physics.f0,
        #     )
        # )
        # var_dict[TotalEnergy.get_name()] = TotalEnergy(
        #     var_dict[TotalKineticEnergy.get_name()],
        #     var_dict[TotalAvailablePotentialEnergy.get_name()],
        # )
