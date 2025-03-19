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
    QGPressure,
    StreamFunction,
    StreamFunctionFromVorticity,
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
from qgsw.masks import Masks
from qgsw.models.names import ModelName
from qgsw.models.qg.projected.modified.variables import (
    RefStreamFunctionFromVorticity,
)
from qgsw.models.qg.projected.projectors.core import QGProjector
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import DEVICE, defaults
from qgsw.utils.units._units import Unit

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
            var_dict (dict[str, DiagnosticVariable]): Variables dictionary.
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
        space_2d = SpaceDiscretization2D.from_config(space)

        var_dict[QGPressure.get_name()] = QGPressure(
            QGProjector(
                A=compute_A(
                    model.h,
                    model.g_prime,
                    **specs,
                ),
                H=model.h.unsqueeze(-1).unsqueeze(-1),
                space=space_2d.add_h(
                    Coordinates1D(points=model.h, unit=Unit.M),
                ),
                f0=physics.f0,
                masks=Masks.empty(space.nx, space.ny, specs["device"]),
            ),
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
        cls.add_pressure(var_dict, model, space, physics)
        cls.add_vorticity(var_dict, space, model, physics)
        cls.add_streamfunction(var_dict, physics, space, model)
        cls.add_enstrophy(var_dict)
        cls.add_energy(var_dict, space, model, physics)

        return var_dict


class RefQGVariableSet(QGVariableSet):
    """Reference QG Variable set for one-mayer models."""

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
        var_dict[RefStreamFunctionFromVorticity.get_name()] = (
            RefStreamFunctionFromVorticity(
                var_dict[PhysicalVorticity.get_name()],
                space.nx,
                space.ny,
                space.dx,
                space.dy,
            )
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
        cls.add_prognostics(var_dict)
        cls.add_physical(var_dict, space)
        cls.add_fluxes(var_dict, space)
        cls.add_pressure(var_dict, model, space, physics)
        cls.add_vorticity(var_dict, space, model, physics)
        cls.add_streamfunction(var_dict, physics, space, model)
        cls.add_enstrophy(var_dict)
        cls.add_energy(var_dict, space, model, physics)

        return var_dict
