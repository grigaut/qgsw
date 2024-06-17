"""Model Configuration."""

from typing import Any, ClassVar

import torch

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError
from qgsw.specs import DEVICE


class ModelConfig(_Config):
    """Model configuration."""

    _colinearity_allowed: ClassVar[list[str]] = [
        "QGColinearSublayerStreamFunction",
    ]

    section: str = keys.MODELS["section"]
    section_several: str = keys.MODELS["section several"]
    _type: str = keys.MODELS["type"]
    _name: str = keys.MODELS["name"]
    _h: str = keys.MODELS["layers"]
    _g_prime: str = keys.MODELS["reduced gravity"]
    _prefix: str = keys.MODELS["prefix"]
    _alpha: str = keys.MODELS["colinearity coef"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate ModelConfig.

        Args:
            params (dict[str, Any]): Script Configuration dictionnary.
        """
        super().__init__(params)

    @property
    def type(self) -> str:
        """Model type."""
        return self.params[self._type]

    @property
    def name(self) -> str:
        """Model name."""
        return self.params[self._name]

    @property
    def name_sc(self) -> str:
        """Model name in Snake Case."""
        return self.name.lower().replace(" ", "_")

    @property
    def h(self) -> torch.Tensor:
        """Values of layers thickness (h)."""
        nl = len(self.params[self._h])
        h = torch.zeros(
            size=(nl, 1, 1),
            dtype=torch.float64,
            device=DEVICE,
        )
        h[:, 0, 0] = torch.Tensor(self.params[self._h])
        return torch.Tensor(self.params[self._h]).to(
            device=DEVICE,
            dtype=torch.float64,
        )

    @property
    def g_prime(self) -> torch.Tensor:
        """Values of reduced gravity (g')."""
        return torch.Tensor(self.params[self._g_prime]).to(
            device=DEVICE,
            dtype=torch.float64,
        )

    @property
    def prefix(self) -> int:
        """Prefix."""
        return self.params[self._prefix]

    @property
    def colinearity_coef(self) -> float:
        """Colinearity Coefficient, only relevant for modified QG models."""
        return self._get_alpha()

    def _get_alpha(self) -> float:
        if self.type not in self._colinearity_allowed:
            msg = (
                "The colinearity coefficient is only relevant "
                f"for the following models: {self._colinearity_allowed}."
            )
            raise AttributeError(msg)
        return self.params[self._alpha]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate that H and g' shapes match.

        Args:
            params (dict[str, Any]): Configuration Parameters.

        Raises:
            ConfigError: H and g' shapes don't match.

        Returns:
            dict[str, Any]: Layers Configuration.
        """
        h_shape = len(params[self._h])
        g_prime_shape = len(params[self._g_prime])

        if h_shape != g_prime_shape:
            msg = (
                f"H shape ({h_shape}) and "
                f"g' 's shape ({g_prime_shape}) don't match."
            )
            raise ConfigError(msg)

        return super()._validate_params(params)
