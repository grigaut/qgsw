"""Data Loaders."""

from __future__ import annotations

import urllib.request
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch

from qgsw.configs.base import _Config, _DataConfig
from qgsw.configs.bathymetry import BathyDataConfig
from qgsw.configs.core import ScriptConfig
from qgsw.data.preprocessing import (
    BathyPreprocessor,
    Preprocessor,
    WindStressPreprocessorSpeed,
    WindStressPreprocessorTaux,
    _WindStressPreprocessor,
)
from qgsw.data.readers import Reader

if TYPE_CHECKING:
    from qgsw.configs.windstress import WindStressDataConfig

Data = TypeVar("Data")
Preprocess = TypeVar("Preprocess", bound=Preprocessor)
Config = TypeVar("Config", bound=_Config)


class Loader(Generic[Config, Data, Preprocess], metaclass=ABCMeta):
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
            config (ScriptConfig): Data Configuration.

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


BathyData = tuple[np.ndarray, np.ndarray, np.ndarray]


class BathyLoader(Loader[BathyDataConfig, BathyData, BathyPreprocessor]):
    """Bathymetry loader."""

    def set_preprocessor(self, config: BathyDataConfig) -> BathyPreprocessor:
        """Set Bathymetric data preprocessor.

        Returns:
            BathyPreprocessor: Preprocessor.
        """
        return BathyPreprocessor(
            longitude_key=config.longitude,
            latitude_key=config.latitude,
            bathymetry_key=config.elevation,
        )

    def _set_config(self, config: BathyDataConfig) -> BathyDataConfig:
        return config


class WindForcingLoader(
    Loader[
        ScriptConfig,
        tuple[torch.Tensor, torch.Tensor],
        _WindStressPreprocessor,
    ]
):
    """Wind Forcing Data Loader."""

    def set_preprocessor(
        self, config: ScriptConfig
    ) -> _WindStressPreprocessor:
        """Set WindStress preprocessor.

        Args:
            config (ScriptConfig): configuration.

        Raises:
            KeyError: If the configuration is not valid.

        Returns:
            _WindStressPreprocessor: Preprocessor.
        """
        ws_data = config.windstress.data
        if ws_data.data_type == "speed":
            return WindStressPreprocessorSpeed(
                longitude_key=ws_data.longitude,
                latitude_key=ws_data.latitude,
                time_key=ws_data.time,
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
            )
        if ws_data.data_type == "tau":
            return WindStressPreprocessorTaux(
                longitude_key=ws_data.longitude,
                latitude_key=ws_data.latitude,
                time_key=ws_data.time,
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
            )
        msg = "Unrecognized data type in windstress.data section."
        raise KeyError(msg)

    def _set_config(self, config: ScriptConfig) -> WindStressDataConfig:
        """Set Data Configuration.

        Args:
            config (ScriptConfig): Script Configuration.

        Returns:
            WindStressDataConfig: Data configuration.
        """
        return config.windstress.data
