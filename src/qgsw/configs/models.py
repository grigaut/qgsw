"""Model Configuration."""

from typing import Any

import torch

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError
from qgsw.specs import DEVICE


class ModelConfig(_Config):
    """Model configuration."""

    _type: str = "type"
    _name: str = "name"
    _h: str = keys.LAYERS["layer thickness"]
    _g_prime: str = keys.LAYERS["reduced gravity"]

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
        h = torch.zeros(
            size=(self.nl, 1, 1),
            dtype=torch.float64,
            device=DEVICE,
        )
        h[:, 0, 0] = torch.Tensor(self.params[self._h])
        return torch.Tensor(self.params[self._h], device=DEVICE).to(
            dtype=torch.float64
        )

    @property
    def g_prime(self) -> torch.Tensor:
        """Values of reduced gravity (g')."""
        return torch.Tensor(self.params[self._g_prime], device=DEVICE).to(
            dtype=torch.float64
        )

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self._nl

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
        self._nl = h_shape

        return super()._validate_params(params)