"""Input-Output Configuration Tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError


class IOConfig(_Config):
    """Input-Output configuration."""

    _name: str = keys.IO["name"]
    _log: str = keys.IO["log performance"]
    _res_section: str = keys.OUTPUT["section"]
    _plots_section: str = keys.PLOTS["section"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate IOConfig.

        Args:
            params (dict[str, Any]): IO configuration dictionnary.
        """
        super().__init__(params)
        self._res = OutputsConfig(self.params[self._res_section])
        self._plots = PlotsConfig(self.params[self._plots_section])

    @property
    def name(self) -> str:
        """Run name."""
        return self._params[self._name]

    @property
    def name_sc(self) -> str:
        """Snake-cased name."""
        return self.name.lower().replace(" ", "_")

    @property
    def results(self) -> OutputsConfig:
        """Results saving configuration."""
        return self._res

    @property
    def plots(self) -> PlotsConfig:
        """Plots configuration."""
        return self._plots

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
        # Verify that the result section is present.
        if self._res_section not in params:
            msg = (
                "The io configuration must contain a "
                f"result section, named {self._res_section}."
            )
            raise ConfigError(msg)
        # Verify that the plots section is present.
        if self._plots_section not in params:
            msg = (
                "The io configuration must contain a "
                f"plots section, named {self._plots_section}."
            )
            raise ConfigError(msg)
        return super()._validate_params(params)


class PlotsConfig(_Config):
    """Plots Configuration."""

    _save: str = keys.PLOTS["save"]
    _show: str = keys.PLOTS["show"]
    _dir: str = keys.PLOTS["directory"]
    _quantity: str = keys.PLOTS["quantity"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate plots configuration object.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        super().__init__(params)
        if self.save and not self.directory.is_dir():
            self.directory.mkdir()
            gitignore = self.directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")

    @property
    def save(self) -> bool:
        """Whether to save the plots or not."""
        return self.params[self._save]

    @property
    def show(self) -> bool:
        """Whether to show the plots or not."""
        return self.params[self._show]

    @property
    def quantity(self) -> int:
        """Number of plots."""
        return self.params[self._quantity]

    @property
    def directory(self) -> Path:
        """Directory in which to save the plots."""
        return Path(self.params[self._dir])

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)


class OutputsConfig(_Config):
    """Plots Configuration."""

    _save: str = keys.OUTPUT["save"]
    _dir: str = keys.OUTPUT["directory"]
    _quantity: str = keys.OUTPUT["quantity"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate plots configuration object.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        super().__init__(params)
        if self.save and not self.directory.is_dir():
            self.directory.mkdir()
            gitignore = self.directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")

    @property
    def save(self) -> bool:
        """Whether to save the plots or not."""
        return self.params[self._save]

    @property
    def directory(self) -> Path:
        """Directory in which to save the plots."""
        return Path(self.params[self._dir])

    @property
    def quantity(self) -> int:
        """Number of outputs."""
        return self.params[self._quantity]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)
