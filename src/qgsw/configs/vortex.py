"""Vortex-related configuration."""

from __future__ import annotations

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config


class VortexConfig(_Config):
    """Vortex Configuration."""

    section: str = keys.VORTEX["section"]
    _type: str = keys.VORTEX["type"]
    _perturbation: str = keys.VORTEX["perturbation magnitude"]

    @property
    def type(self) -> str:
        """Vortex Type."""
        return self.params[self._type]

    @property
    def perturbation_magnitude(self) -> str:
        """Vortex perturbation magnitude."""
        return self.params[self._perturbation]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)
