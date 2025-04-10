"""SW variable set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    LayerDepthAnomalyDiag,
    MeridionalVelocityDiag,
    PhysicalVorticity,
    Psi2,
    StreamFunctionFromVorticity,
    TimeDiag,
    Vorticity,
    ZonalVelocityDiag,
)

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class SWVariableSet:
    """SW Variable set."""

    @classmethod
    def add_prognostics(
        cls,
        var_dict: dict[str, DiagnosticVariable],
    ) -> None:
        """Add prognostic variables.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dict.
        """
        var_dict[TimeDiag.get_name()] = TimeDiag()
        var_dict[ZonalVelocityDiag.get_name()] = ZonalVelocityDiag()
        var_dict[MeridionalVelocityDiag.get_name()] = MeridionalVelocityDiag()
        var_dict[LayerDepthAnomalyDiag.get_name()] = LayerDepthAnomalyDiag()

    @classmethod
    def add_physical(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
    ) -> None:
        """Add physical variables.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dict.
            space (SpaceConfig): Space configuration. configuration.
        """

    @classmethod
    def add_fluxes(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
    ) -> None:
        """Add fluxes.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            space (SpaceConfig): Space configuration.
        """

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

    @classmethod
    def add_streamfunction(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
    ) -> None:
        """Add streamfunction.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            physics (PhysicsConfig): Physics Configuration.
            space (PhysicsConfig): Space Configuration.
            model (ModelConfig): Model Configuration, for compatibility only.
        """
        var_dict[StreamFunctionFromVorticity.get_name()] = (
            StreamFunctionFromVorticity(
                var_dict[PhysicalVorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
            )
        )
        var_dict[Psi2.get_name()] = Psi2(
            var_dict[StreamFunctionFromVorticity.get_name()],
        )

    @classmethod
    def add_vorticity(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
    ) -> None:
        """Add vorticity.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            space (SpaceConfig): Space configuration.
            model (ModelConfig): Model Configuration.
            physics (PhysicsConfig): Physics Configuration.
        """
        var_dict[Vorticity.get_name()] = Vorticity()
        var_dict[PhysicalVorticity.get_name()] = PhysicalVorticity(
            var_dict[Vorticity.get_name()],
            space.ds,
        )

    @classmethod
    def add_enstrophy(cls, var_dict: dict[str, DiagnosticVariable]) -> None:
        """Add enstrophy.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
        """

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
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            space (SpaceConfig): Space configuration.
            model (ModelConfig): Model Configuration.
            physics (PhysicsConfig): Physics Configuration.
        """

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,
        physics: PhysicsConfig,  # noqa: ARG003
        model: ModelConfig,  # noqa: ARG003
    ) -> dict[str, DiagnosticVariable]:
        """Create variable set.

        Args:
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
            model (ModelConfig): Model configuaration.

        Returns:
            dict[str, DiagnosticVariable]: Variables dictionnary.
        """
        var_dict = {}
        cls.add_prognostics(var_dict)
        cls.add_vorticity(var_dict, space)
        cls.add_streamfunction(var_dict, space)

        return var_dict
