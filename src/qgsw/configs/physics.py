"""Physics-Realted Configurations."""

from __future__ import annotations

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError


class PhysicsConfig(_Config):
    """Physics Configuration."""

    _rho: str = keys.PHYSICS["rho"]
    _slip_coef: str = keys.PHYSICS["slip coef"]
    _coriolis_param: str = keys.PHYSICS["f0"]
    _beta: str = keys.PHYSICS["beta"]
    _wstress_mag: str = keys.PHYSICS["wind stress magnitude"]
    _drag_coef: str = keys.PHYSICS["drag coefficient"]

    @property
    def slip_coef(self) -> float:
        """Slip coefficient value."""
        return self.params[self._slip_coef]

    @property
    def drag_coefficient(self) -> float:
        """Surface drag coefficient."""
        return self.params[self._drag_coef]

    @property
    def rho(self) -> float:
        """Density."""
        return self.params[self._rho]

    @property
    def f0(self) -> float:
        """Coriolis Parameter."""
        return self.params[self._coriolis_param]

    @property
    def beta(self) -> float:
        """Beta (from beta-plane approximation) value."""
        return self.params[self._beta]

    @property
    def bottom_drag_coef(self) -> float:
        """Drag Coefficient."""
        return 0.5 * self.f0 * 2.0 / 2600  # Source ?

    @property
    def wind_stress_magnitude(self) -> float:
        """Wind Stress Magnitude (in Pa m-1 kg s-2)."""
        return self.params[self._wstress_mag]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Physics Configuration.

        Args:
            params (dict[str, Any]): Physics configuration.

        Returns:
            dict[str, Any]: Physics Configuration.
        """
        # Verify slip coefficient value
        slip_coef = params[self._slip_coef]
        if (slip_coef < 0) or (slip_coef > 1):
            msg = f"Slip coefficient must be in [0, 1], got {slip_coef}."
            raise ConfigError(msg)

        return super()._validate_params(params)
