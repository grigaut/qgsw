"""VariableSet related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    Pressure,
    SurfaceHeightAnomaly,
)
from qgsw.models.qg.projected.modified.filtered.pv import (
    compute_g_tilde,
)
from qgsw.models.qg.projected.variable_set import QGVariableSet

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGCollinearFilteredSFVariableSet(QGVariableSet):
    """Variable set for QGCOllinearFilteredSF."""

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
        var_dict[SurfaceHeightAnomaly.get_name()] = SurfaceHeightAnomaly()
        eta_phys_name = PhysicalSurfaceHeightAnomaly.get_name()
        var_dict[eta_phys_name] = PhysicalSurfaceHeightAnomaly(
            var_dict[PhysicalLayerDepthAnomaly.get_name()],
        )
        var_dict[Pressure.get_name()] = Pressure(
            compute_g_tilde(
                model.g_prime,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1),
            var_dict[eta_phys_name],
        )
