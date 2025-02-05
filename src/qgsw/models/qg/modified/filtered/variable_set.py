"""VariableSet related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    PhysicalVorticity,
    Pressure,
    SurfaceHeightAnomaly,
    Vorticity,
)
from qgsw.filters.spectral import SpectralGaussianFilter2D
from qgsw.models.qg.modified.filtered.pv import (
    CollinearFilteredPotentialVorticity,
    compute_g_tilde,
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
            H=model.h,
            g_prime=model.g_prime,
            f0=physics.f0,
            ds=space.ds,
            filt=SpectralGaussianFilter2D(model.sigma),
        )
