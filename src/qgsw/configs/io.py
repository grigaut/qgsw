"""Input-Output Configuration Tools."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.utils.storage import get_absolute_storage_path


class IOConfig(_Config):
    """Input-Output configuration."""

    section: str = keys.IO["section"]
    _name: str = keys.IO["name"]
    _log: str = keys.IO["log performance"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate IOConfig.

        Args:
            params (dict[str, Any]): IO configuration dictionnary.
        """
        super().__init__(params)
        self._res = OutputsConfig.parse(params)

    @property
    def name(self) -> str:
        """Run name."""
        return self._params[self._name]

    @property
    def name_sc(self) -> str:
        """Snake-cased name."""
        return self.name.lower().replace(" ", "_")

    @cached_property
    def results(self) -> OutputsConfig:
        """Results saving configuration."""
        return OutputsConfig.parse(self.params)

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


class OutputsConfig(_Config):
    """Plots Configuration."""

    section: str = keys.OUTPUT["section"]
    _save: str = keys.OUTPUT["save"]
    _dir: str = keys.OUTPUT["directory"]
    _quantity: str = keys.OUTPUT["quantity"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate plots configuration object.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        super().__init__(params)

    @property
    def save(self) -> bool:
        """Whether to save the plots or not."""
        return self.params[self._save]

    @property
    def directory(self) -> Path:
        """Directory in which to save the results."""
        directory = get_absolute_storage_path(Path(self.params[self._dir]))
        if self.save and not directory.is_dir():
            directory.mkdir()
            gitignore = directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")
        return directory

    @property
    def quantity(self) -> int:
        """Number of outputs."""
        return self.params[self._quantity]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)
