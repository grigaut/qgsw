"""Physics-Realted Configurations."""

from __future__ import annotations

from typing import Any

import numpy as np

from qgsw import verbose
from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError


class PhysicsConfig(_Config):
    """Physics Configuration."""

    section: str = keys.PHYSICS["section"]
    _rho: str = keys.PHYSICS["rho"]
    _slip_coef: str = keys.PHYSICS["slip coef"]
    _coriolis_param: str = keys.PHYSICS["f0"]
    _beta: str = keys.PHYSICS["beta"]
    _bottom_drag: str = keys.PHYSICS["bottom drag coefficient"]

    @property
    def slip_coef(self) -> float:
        """Slip coefficient value."""
        return self.params[self._slip_coef]

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
        if np.isnan(self.params[self._bottom_drag]):
            verbose.display(
                msg="Bottom drag coefficient inferred using f0.",
                trigger_level=1,
            )
            return 0.5 * self.f0 * 2.0 / 2600  # Source ?
        return self.params[self._bottom_drag]

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
