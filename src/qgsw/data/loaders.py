"""Data Loaders."""

from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from qgsw.configs.base import _Config, _DataConfig
from qgsw.data.preprocessing import Preprocessor
from qgsw.data.readers import Reader

Data = TypeVar("Data")
Preprocess = TypeVar("Preprocess", bound=Preprocessor)
Config = TypeVar("Config", bound=_Config)


class Loader(ABC, Generic[Config, Data, Preprocess]):
    """Data loader."""

    def __init__(self, config: Config) -> None:
        """Instantiate Loader.

        Args:
            config (Config): Loader data configuration.
        """
        self._config = self._set_config(config=config)
        filepath = self._set_filepath()
        self._load_from_url(filepath=filepath)
        self._reader = Reader(filepath=filepath)
        self._preprocess = self.set_preprocessor(config=config)

    @abstractmethod
    def set_preprocessor(self, config: Config) -> Preprocess:
        """Instantiate preprocessor."""

    def _set_filepath(self) -> Path:
        """Set filepath.

        Args:
            config (RunConfig): Data Configuration.

        Returns:
            Path: Path to save data at.
        """
        filename = Path(self._config.url).name
        return self._config.folder.joinpath(filename)

    @abstractmethod
    def _set_config(self, config: Config) -> _DataConfig:
        """Set filepath."""

    def retrieve(self) -> Data:
        """Retrieve Data.

        Returns:
            Data: Data.
        """
        return self._preprocess(self._reader)

    def _load_from_url(self, filepath: Path) -> None:
        """Create BathyLoader from a URL.

        Args:
            filepath (Path): Filepath to save data in.

        Returns:
            Self: Instantiated BathyLoader.
        """
        if not filepath.is_file():
            print(f"Downloading file {filepath} from {self._config.url}...")
            urllib.request.urlretrieve(self._config.url, filepath)
            print("..done")
