"""VariableSet related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.dynamics import PhysicalVorticity, Vorticity
from qgsw.models.qg.modified.filtered.pv import (
    CollinearFilteredPotentialVorticity,
)
from qgsw.models.qg.variable_set import QGVariableSet

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGCollinearFilteredSFVariableSet(QGVariableSet):
    """Variable set for QGCOllinearFilteredSF."""

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
        cf_pv_name = CollinearFilteredPotentialVorticity.get_name()
        var_dict[cf_pv_name] = CollinearFilteredPotentialVorticity(
            model.h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * space.ds,
            space.ds,
            physics.f0,
        )
