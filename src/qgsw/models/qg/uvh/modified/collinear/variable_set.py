"""Variable set related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.physical import (
    CollinearityCoefficientDiag,
    LayerDepthAnomaly,
    MeridionalVelocity,
    MeridionalVelocityFromPsi2,
    QGPressure,
    StreamFunction,
    StreamFunctionFromVorticity,
    SurfaceHeightAnomaly,
    TimeDiag,
    Vorticity,
    ZonalVelocity,
    ZonalVelocityFromPsi2,
)
from qgsw.models.qg.uvh.modified.collinear.variables import (
    CollinearPsi2,
)
from qgsw.models.qg.uvh.projectors.collinear import CollinearSFProjector
from qgsw.models.qg.uvh.variable_set import QGVariableSet

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGCollinearSFVariableSet(QGVariableSet):
    """QgCollinearSF Variable Set."""

    @classmethod
    def add_physical(
        cls,
        var_dict: dict[str, DiagnosticVariable],
    ) -> None:
        """Add physical variables.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dict.
            space (SpaceConfig): Space configuration. configuration.
        """
        var_dict[TimeDiag.get_name()] = TimeDiag()
        var_dict[ZonalVelocity.get_name()] = ZonalVelocity()
        var_dict[MeridionalVelocity.get_name()] = MeridionalVelocity()
        var_dict[LayerDepthAnomaly.get_name()] = LayerDepthAnomaly()
        var_dict[CollinearityCoefficientDiag.get_name()] = (
            CollinearityCoefficientDiag()
        )

    @classmethod
    def add_pressure(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        model: ModelConfig,
        space: SpaceConfig,
        physics: PhysicsConfig,
    ) -> None:
        """Add pressure.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            model (ModelConfig): Model Configuration.
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
        """
        var_dict[SurfaceHeightAnomaly.get_name()] = SurfaceHeightAnomaly()
        P = CollinearSFProjector.from_config(space, model, physics)  # noqa: N806
        var_dict[QGPressure.get_name()] = QGPressure(P, space.dx, space.dy)

    @classmethod
    def add_streamfunction(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        physics: PhysicsConfig,
        space: SpaceConfig,
        model: ModelConfig,  # noqa: ARG003
    ) -> None:
        """Add streamfunction.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            physics (PhysicsConfig): Physics Configuration.
            space (PhysicsConfig): Space Configuration.
            model (ModelConfig): Model Configuration, for compatibility only.
        """
        var_dict[StreamFunction.get_name()] = StreamFunction(
            var_dict[QGPressure.get_name()],
            physics.f0,
        )
        var_dict[StreamFunctionFromVorticity.get_name()] = (
            StreamFunctionFromVorticity(
                var_dict[Vorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
            )
        )
        var_dict[CollinearPsi2.get_name()] = CollinearPsi2(
            var_dict[StreamFunctionFromVorticity.get_name()],
        )
        var_dict[ZonalVelocityFromPsi2.get_name()] = ZonalVelocityFromPsi2(
            var_dict[CollinearPsi2.get_name()],
            space.dy,
        )
        var_dict[MeridionalVelocityFromPsi2.get_name()] = (
            MeridionalVelocityFromPsi2(
                var_dict[CollinearPsi2.get_name()],
                space.dx,
            )
        )
