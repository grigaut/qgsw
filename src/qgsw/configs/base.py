"""Base Configuration."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

import toml
from typing_extensions import Self


class _Config(metaclass=ABCMeta):
    """Configuration."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate configuration from configuration parameters dictionnary.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        self._params = self._validate_params(params=params)

    @property
    def params(self) -> dict[str, Any]:
        """Configuration parameters dictionnary."""
        return self._params

    def __repr__(self) -> str:
        """Representation of the configuration.

        Returns:
            str: Configuration params.
        """
        return self.params.__repr__()

    @abstractmethod
    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate prameters values."""
        return params

    @classmethod
    def from_file(cls, config_path: Path) -> Self:
        """Instantiate Parser from configuration filepath.

        Args:
            config_path (Path): Configuration file path.

        Returns:
            Self: Parser.
        """
        return cls(params=toml.load(config_path))


class _DataConfig(_Config):
    """Data Configuration."""

    _url: str
    _folder: str

    @property
    def url(self) -> str:
        """Data URL."""
        return self.params[self._url]

    @property
    def folder(self) -> Path:
        """Data saving folder."""
        return Path(self.params[self._folder])
