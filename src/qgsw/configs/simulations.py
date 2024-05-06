"""Run configuration."""

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config


class SimulationConfig(_Config):
    """Simulation configuration."""

    section: str = keys.SIMULATION["section"]
    _duration: str = keys.SIMULATION["duration"]
    _dt: str = keys.SIMULATION["timestep"]

    @property
    def dt(self) -> float:
        """Timestep."""
        return self.params[self._dt]

    @property
    def duration(self) -> float:
        """Simulation duration, relative unit."""
        return self.params[self._duration]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)
