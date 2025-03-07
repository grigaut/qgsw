"""QG variable set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.dynamics import (
    Enstrophy,
    LayerDepthAnomalyDiag,
    MeridionalVelocityDiag,
    MeridionalVelocityFlux,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalSurfaceHeightAnomaly,
    PhysicalVorticity,
    PhysicalZonalVelocity,
    PotentialVorticity,
    Pressure,
    StreamFunction,
    SurfaceHeightAnomaly,
    TimeDiag,
    TotalEnstrophy,
    Vorticity,
    ZonalVelocityDiag,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.energetics import (
    ModalAvailablePotentialEnergy,
    ModalEnergy,
    ModalKineticEnergy,
    TotalAvailablePotentialEnergy,
    TotalEnergy,
    TotalKineticEnergy,
)
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGVariableSet:
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
        var_dict[PhysicalZonalVelocity.get_name()] = PhysicalZonalVelocity(
            space.dx,
        )
        var_dict[PhysicalMeridionalVelocity.get_name()] = (
            PhysicalMeridionalVelocity(space.dy)
        )
        var_dict[PhysicalLayerDepthAnomaly.get_name()] = (
            PhysicalLayerDepthAnomaly(
                space.ds,
            )
        )

    @classmethod
    def add_fluxes(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        space: SpaceConfig,
    ) -> None:
        """Add fluxes.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
            space (SpaceConfig): Space configuration.
        """
        var_dict[ZonalVelocityFlux.get_name()] = ZonalVelocityFlux(space.dx)
        var_dict[MeridionalVelocityFlux.get_name()] = MeridionalVelocityFlux(
            space.dy,
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
        var_dict[SurfaceHeightAnomaly.get_name()] = SurfaceHeightAnomaly()
        eta_phys_name = PhysicalSurfaceHeightAnomaly.get_name()
        var_dict[eta_phys_name] = PhysicalSurfaceHeightAnomaly(
            var_dict[PhysicalLayerDepthAnomaly.get_name()],
        )
        var_dict[Pressure.get_name()] = Pressure(
            model.g_prime.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            var_dict[eta_phys_name],
        )

    @classmethod
    def add_streamfunction(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        physics: PhysicsConfig,
    ) -> None:
        """Add streamfunction.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
            physics (PhysicsConfig): Physics Confdiguration.
        """
        var_dict[StreamFunction.get_name()] = StreamFunction(
            var_dict[Pressure.get_name()],
            physics.f0,
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
            model.h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            space.ds,
            physics.f0,
        )

    @classmethod
    def add_enstrophy(cls, var_dict: dict[str, DiagnosticVariable]) -> None:
        """Add enstrophy.

        Args:
            var_dict (dict[str, DiagnosticVariable]): _description_
        """
        var_dict[Enstrophy.get_name()] = Enstrophy(
            var_dict[PotentialVorticity.get_name()],
        )
        var_dict[TotalEnstrophy.get_name()] = TotalEnstrophy(
            var_dict[PotentialVorticity.get_name()],
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
        A = compute_A(  # noqa: N806
            model.h,
            model.g_prime,
            torch.float64,
            DEVICE.get(),
        )
        var_dict[ModalKineticEnergy.get_name()] = ModalKineticEnergy(
            A,
            var_dict[StreamFunction.get_name()],
            model.h,
            space.dx,
            space.dy,
        )
        var_dict[ModalAvailablePotentialEnergy.get_name()] = (
            ModalAvailablePotentialEnergy(
                A,
                var_dict[StreamFunction.get_name()],
                model.h,
                physics.f0,
            )
        )
        var_dict[ModalEnergy.get_name()] = ModalEnergy(
            var_dict[ModalKineticEnergy.get_name()],
            var_dict[ModalAvailablePotentialEnergy.get_name()],
        )
        var_dict[TotalKineticEnergy.get_name()] = TotalKineticEnergy(
            var_dict[StreamFunction.get_name()],
            model.h,
            space.dx,
            space.dy,
        )
        var_dict[TotalAvailablePotentialEnergy.get_name()] = (
            TotalAvailablePotentialEnergy(
                A,
                var_dict[StreamFunction.get_name()],
                model.h,
                physics.f0,
            )
        )
        var_dict[TotalEnergy.get_name()] = TotalEnergy(
            var_dict[TotalKineticEnergy.get_name()],
            var_dict[TotalAvailablePotentialEnergy.get_name()],
        )

    @classmethod
    def get_variable_set(
        cls,
        space: SpaceConfig,
        physics: PhysicsConfig,
        model: ModelConfig,
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
        cls.add_physical(var_dict, space)
        cls.add_fluxes(var_dict, space)
        cls.add_pressure(var_dict, model)
        cls.add_streamfunction(var_dict, physics)
        cls.add_vorticity(var_dict, space, model, physics)
        cls.add_enstrophy(var_dict)
        cls.add_energy(var_dict, space, model, physics)

        return var_dict
