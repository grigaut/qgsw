"""Variable set related to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    CollinearityCoefficientDiag,
    PhysicalVorticity,
    PotentialVorticity,
    QGPressure,
    StreamFunction,
    StreamFunctionFromVorticity,
    Vorticity,
)
from qgsw.masks import Masks
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.modified.collinear.variables import (
    CollinearPsi2,
)
from qgsw.models.qg.uvh.modified.filtered.pv import compute_g_tilde
from qgsw.models.qg.uvh.projectors.collinear import CollinearQGProjector
from qgsw.models.qg.uvh.variable_set import QGVariableSet
from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import defaults
from qgsw.utils.units._units import Unit

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
        super().add_pressure(var_dict, model, space, physics)

        specs = defaults.get()
        space_2d = SpaceDiscretization2D.from_config(space)
        P = CollinearQGProjector(  # noqa: N806
            A=compute_A(
                H=model.h[:1],
                g_prime=compute_g_tilde(model.g_prime),
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

        var_dict[QGPressure.get_name()] = QGPressure(
            P,
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
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
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
                var_dict[PhysicalVorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
            )
        )
        var_dict[CollinearPsi2.get_name()] = CollinearPsi2(
            var_dict[StreamFunctionFromVorticity.get_name()],
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
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
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
