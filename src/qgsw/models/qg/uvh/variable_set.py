"""QG variable set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.physical import (
    Enstrophy,
    LayerDepthAnomaly,
    MeridionalVelocity,
    MeridionalVelocity2,
    ModalAvailablePotentialEnergy,
    ModalEnergy,
    ModalKineticEnergy,
    PotentialVorticity,
    Pressure,
    Psi2,
    QGPressure,
    RefMeridionalVelocity2,
    RefZonalVelocity2,
    StreamFunction,
    StreamFunctionFromVorticity,
    SurfaceHeightAnomaly,
    Time,
    TimeDiag,
    TotalAvailablePotentialEnergy,
    TotalEnergy,
    TotalEnstrophy,
    TotalKineticEnergy,
    Vorticity,
    ZonalVelocity,
    ZonalVelocity2,
)
from qgsw.models.names import ModelName
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.modified.variables import (
    Psi21L,
)
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.fields.variables.base import DiagnosticVariable


class QGVariableSet:
    """QG Variable set."""

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
        var_dict[ZonalVelocity2.get_name()] = ZonalVelocity2()
        var_dict[MeridionalVelocity2.get_name()] = MeridionalVelocity2()

    @classmethod
    def add_pressure(
        cls,
        var_dict: dict[str, DiagnosticVariable],
        physics: PhysicsConfig,
        space: SpaceConfig,
        model: ModelConfig,
    ) -> None:
        """Add pressure.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            model (ModelConfig): Model Configuration.
            space (SpaceConfig): Space configuration.
            physics (PhysicsConfig): Physics configuration.
        """
        var_dict[SurfaceHeightAnomaly.get_name()] = SurfaceHeightAnomaly()
        var_dict[QGPressure.get_name()] = QGPressure(
            QGProjector.from_config(space, model, physics),
            space.dx,
            space.dy,
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
            var_dict[Pressure.get_name()],
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
        var_dict[Psi2.get_name()] = Psi2(
            var_dict[StreamFunctionFromVorticity.get_name()],
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
            physics (PhysicsConfig): Physics Configuration.
        """
        var_dict[Vorticity.get_name()] = Vorticity(
            space.dx,
            space.dy,
        )
        var_dict[PotentialVorticity.get_name()] = PotentialVorticity(
            model.h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            space.dx,
            space.dy,
            physics.f0,
        )

    @classmethod
    def add_enstrophy(cls, var_dict: dict[str, DiagnosticVariable]) -> None:
        """Add enstrophy.

        Args:
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
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
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
            space (SpaceConfig): Space configuration.
            model (ModelConfig): Model Configuration.
            physics (PhysicsConfig): Physics Configuration.
        """
        A = compute_A(
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
        cls.add_physical(var_dict)
        cls.add_pressure(var_dict, physics, space, model)
        cls.add_vorticity(var_dict, space, model, physics)
        cls.add_streamfunction(var_dict, physics, space, model)
        cls.add_enstrophy(var_dict)
        cls.add_energy(var_dict, space, model, physics)

        return var_dict


class RefQGVariableSet(QGVariableSet):
    """Reference QG Variable set for one-mayer models."""

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
        var_dict[Time.get_name()] = Time()
        var_dict[ZonalVelocity.get_name()] = ZonalVelocity()
        var_dict[MeridionalVelocity.get_name()] = MeridionalVelocity()
        var_dict[LayerDepthAnomaly.get_name()] = LayerDepthAnomaly()
        var_dict[RefZonalVelocity2.get_name()] = RefZonalVelocity2(
            var_dict[ZonalVelocity.get_name()],
        )
        var_dict[RefMeridionalVelocity2.get_name()] = RefMeridionalVelocity2(
            var_dict[MeridionalVelocity.get_name()],
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
                var_dict[Vorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
            )
        )
        var_dict[Psi21L.get_name()] = Psi21L(
            var_dict[StreamFunctionFromVorticity.get_name()],
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
        if model.h.shape[0] != 1 or model.type != ModelName.QUASI_GEOSTROPHIC:
            msg = "Such variable is only suited for one-layer QG models."
            raise ValueError(msg)
        var_dict = {}
        cls.add_physical(var_dict)
        cls.add_pressure(var_dict, physics, space, model)
        cls.add_vorticity(var_dict, space, model, physics)
        cls.add_streamfunction(var_dict, physics, space, model)
        cls.add_enstrophy(var_dict)
        cls.add_energy(var_dict, space, model, physics)

        return var_dict
