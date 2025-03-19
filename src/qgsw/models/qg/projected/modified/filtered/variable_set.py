"""VariableSet related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    PhysicalLayerDepthAnomaly,
    PhysicalSurfaceHeightAnomaly,
    PhysicalVorticity,
    QGPressure,
    StreamFunction,
    SurfaceHeightAnomaly,
)
from qgsw.masks import Masks
from qgsw.models.qg.projected.modified.filtered.pv import (
    compute_g_tilde,
)
from qgsw.models.qg.projected.modified.filtered.variables import (
    ColFiltStreamFunctionFromVorticity,
)
from qgsw.models.qg.projected.projectors.filtered import (
    CollinearFilteredQGProjector,
)
from qgsw.models.qg.projected.variable_set import QGVariableSet
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import defaults
from qgsw.utils.units._units import Unit

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
        specs = defaults.get()
        var_dict[SurfaceHeightAnomaly.get_name()] = SurfaceHeightAnomaly()
        eta_phys_name = PhysicalSurfaceHeightAnomaly.get_name()
        var_dict[eta_phys_name] = PhysicalSurfaceHeightAnomaly(
            var_dict[PhysicalLayerDepthAnomaly.get_name()],
        )
        g_tilde = compute_g_tilde(model.g_prime)
        space_2d = SpaceDiscretization2D.from_config(space)
        P = CollinearFilteredQGProjector(  # noqa: N806
            A=compute_A(
                model.h[:1],
                g_tilde,
                **specs,
            ),
            H=model.h.unsqueeze(-1).unsqueeze(-1),
            g_prime=model.g_prime.unsqueeze(-1).unsqueeze(-1),
            space=space_2d.add_h(
                Coordinates1D(points=model.h[:1], unit=Unit.M),
            ),
            f0=physics.f0,
            masks=Masks.empty(space.nx, space.ny, specs["device"]),
        )
        P.filter.sigma = model.sigma
        var_dict[QGPressure.get_name()] = QGPressure(
            P,
        )

    @classmethod
    def add_streamfunction(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        physics: PhysicsConfig,
        space: SpaceConfig,
        model: ModelConfig,
    ) -> None:
        """Add streamfunction.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            physics (PhysicsConfig): Physics Configuration.
            space (PhysicsConfig): Space Configuration.
            model (ModelConfig): Model Configuration.
        """
        var_dict[StreamFunction.get_name()] = StreamFunction(
            var_dict[QGPressure.get_name()],
            physics.f0,
        )
        var_dict[ColFiltStreamFunctionFromVorticity.get_name()] = (
            ColFiltStreamFunctionFromVorticity(
                var_dict[PhysicalVorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
                filt=CollinearFilteredQGProjector.create_filter(model.sigma),
            )
        )
