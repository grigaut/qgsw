"""Input-Output Configuration Tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config


class IOConfig(_Config):
    """Input-Output configuration."""

    _name: str = keys.IO["name"]
    _output: str = keys.IO["output directory"]
    _freq: str = keys.IO["plot frequency"]
    _log: str = keys.IO["log performance"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate IOConfig.

        Args:
            params (dict[str, Any]): IO configuration dictionnary.
        """
        super().__init__(params)
        if not self.output_directory.is_dir():
            self.output_directory.mkdir()
            gitignore = self.output_directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")

    @property
    def name(self) -> str:
        """Run name."""
        return self._params[self._name]

    @property
    def name_sc(self) -> str:
        """Snake-cased name."""
        return self.name.lower().replace(" ", "_")

    @property
    def output_directory(self) -> Path:
        """Output directory."""
        return Path(self._params[self._output])

    @property
    def plot_bool(self) -> bool:
        """Whether to plot during run or not."""
        return self._params[self._freq] is not None

    @property
    def plot_frequency(self) -> float:
        """Plot frequency, 0 is equivalent to no plotting."""
        if not self.plot_bool:
            return 0
        return self._params[self._freq]

    @property
    def log_performance(self) -> bool:
        """Whether to log performances or not."""
        return self._params[self._log]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate IO paramaters.

        Args:
            params (dict[str, Any]): IO configuration dictionnary.

        Returns:
            dict[str, Any]: IO configuration dictionnary.
        """
        return super()._validate_params(params)
