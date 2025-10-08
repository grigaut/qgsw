"""Data Loaders."""

from __future__ import annotations

import urllib.request
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Union

from qgsw.logging import getLogger

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch

from qgsw.configs.bathymetry import BathyDataConfig
from qgsw.configs.windstress import WindStressConfig, WindStressDataConfig
from qgsw.data.preprocessing import (
    BathyPreprocessor,
    Preprocessor,
    WindStressPreprocessorSpeed,
    WindStressPreprocessorTaux,
    _WindStressPreprocessor,
)
from qgsw.data.readers import Reader

Data = TypeVar("Data")
Preprocess = TypeVar("Preprocess", bound=Preprocessor)
DataConfig = Union[BathyDataConfig, WindStressDataConfig]
Config = TypeVar("Config", bound=DataConfig)
logger = getLogger(__name__)


class Loader(Generic[Config, Data, Preprocess], metaclass=ABCMeta):
    """Data loader."""

    def __init__(
        self,
        url: str,
        output_folder: Path,
        preprocessor: Preprocessor,
    ) -> None:
        """Instantiate Loader.

        Args:
            url (str): URL to retrieve data from.
            output_folder (Path): Folder to store data in.
            preprocessor (Preprocessor): Data Preprocessor.
        """
        self._url = url
        self._folder_out = output_folder
        filepath = self._set_filepath()
        self._load_from_url(filepath=filepath)
        self._reader = Reader(filepath=filepath)
        self._preprocess = preprocessor

    @property
    def url(self) -> str:
        """URL to retrieve from."""
        return self._url

    @property
    def output_folder(self) -> Path:
        """Output folder."""
        return self._folder_out

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
        filename = Path(self.url).name
        return self.output_folder.joinpath(filename)

    @abstractmethod
    def _set_config(self, config: Config) -> DataConfig:
        """Set filepath."""

    def retrieve(self) -> Data:
        """Retrieve Data.

        Returns:
            Data: Data.
        """
        return self._preprocess(self._reader)

    def _load_from_url(self, filepath: Path) -> None:
        """Create Loader from a URL.

        Args:
            filepath (Path): Filepath to save data in.

        Returns:
            Self: Instantiated Loader.
        """
        if not filepath.is_file():
            msg = f"Downloading file {filepath} from {self.url}..."
            logger.info(msg)
            urllib.request.urlretrieve(self.url, filepath)
            logger.info(msg="..done")

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> Self:
        """Instantiate the loader from configuration.

        Args:
            config (Config): Configuration.

        Returns:
            Self: Loader.
        """


BathyData = tuple[np.ndarray, np.ndarray, np.ndarray]


class BathyLoader(Loader[BathyDataConfig, BathyData, BathyPreprocessor]):
    """Bathymetry loader."""

    @classmethod
    def from_config(cls, config: BathyDataConfig) -> Self:
        """Instantiate the BathyLoader from configuration.

        Args:
            config (BathyDataConfig): Configuration.

        Returns:
            Self: BathyLoader
        """
        return cls(
            url=config.url,
            folder=config.folder,
            preprocessor=BathyPreprocessor.from_config(config),
        )


class WindForcingLoader(
    Loader[
        WindStressConfig,
        tuple[torch.Tensor, torch.Tensor],
        _WindStressPreprocessor,
    ],
):
    """Wind Forcing Data Loader."""

    @classmethod
    def from_config(cls, config: WindStressDataConfig) -> Self:
        """Instantiate the WindForcingLoader from configuration.

        Args:
            config (WindStressDataConfig): Configuration.

        Raises:
            KeyError: If the windstress type is not recognized.

        Returns:
            Self: WindForcingLoader
        """
        if config.data_type == "speed":
            preprocessor = WindStressPreprocessorSpeed(
                longitude_key=config.longitude,
                latitude_key=config.latitude,
                time_key=config.time,
                u10_key=config.field_1,
                v10_key=config.field_2,
            )
        elif config.data_type == "tau":
            preprocessor = WindStressPreprocessorTaux(
                longitude_key=config.longitude,
                latitude_key=config.latitude,
                time_key=config.time,
                u10_key=config.field_1,
                v10_key=config.field_2,
            )
        else:
            msg = "Unrecognized data type in windstress.data section."
            raise KeyError(msg)
        return cls(
            url=config.url,
            folder=config.folder,
            preprocessor=preprocessor,
        )
