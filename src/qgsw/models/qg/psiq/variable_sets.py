"""QGPSIQ variable set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.physical import (
    ProgPotentialVorticityDiag,
    ProgStreamFunctionDiag,
    TimeDiag,
)
from qgsw.models.qg.psiq.variables import Psi2

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGPSIQVariableSet:
    """QG Variable set."""

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
        var_dict[ProgPotentialVorticityDiag.get_name()] = (
            ProgPotentialVorticityDiag()
        )
        var_dict[ProgStreamFunctionDiag.get_name()] = ProgStreamFunctionDiag()

    @classmethod
    def add_streamfunction(
        cls,
        var_dict: dict[str, DiagnosticVariable],
    ) -> None:
        """Add streamfunction.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            physics (PhysicsConfig): Physics Configuration.
            space (PhysicsConfig): Space Configuration.
            model (ModelConfig): Model Configuration, for compatibility only.
        """
        var_dict[Psi2.get_name()] = Psi2()

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,  # noqa: ARG003
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
        cls.add_streamfunction(var_dict)

        return var_dict
