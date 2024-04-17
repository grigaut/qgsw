"""Data Loading objects."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import scipy.io
import xarray

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

T = TypeVar("T")


class LoadingError(Exception):
    """Error occuring when loading file."""


class BaseReader(Generic[T], metaclass=ABCMeta):
    """Base class for loaders."""

    def __init__(
        self,
        filepath: Path,
    ) -> None:
        """Instantiate BathyLoader.

        Args:
            filepath: Path: Filepath to data.
        """
        self._filepath = filepath
        self._suffix = self._filepath.suffix
        self._data = self._open()

    @abstractmethod
    def _open(self) -> T:
        """Open data file.

        Returns:
            T: Data
        """

    @abstractmethod
    def get_1d(self, key: str) -> np.ndarray:
        """Access 1D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """

    @abstractmethod
    def get(self, key: str) -> np.ndarray:
        """Access 2D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """


class MatReader(BaseReader[dict]):
    """Data reader for MATLAB files."""

    def _open(self) -> dict:
        """Open MATLAB files.

        Returns:
            dict: MATLAB file content.
        """
        return scipy.io.loadmat(self._filepath)

    def _get(self, key: str) -> np.ndarray:
        """Access data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._data[key]

    def get_1d(self, key: str) -> np.ndarray:
        """Access 1D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._get(key)[:, 0]

    def get(self, key: str) -> np.ndarray:
        """Access 2D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._get(key)


class NCReader(BaseReader[xarray.Dataset]):
    """Data reader for netcdf files."""

    def _open(self) -> xarray.Dataset:
        """Open NetCDF files.

        Returns:
            xarray.Dataset: NetCDF data content.
        """
        return xarray.open_dataset(self._filepath)

    def _get(self, key: str) -> np.ndarray:
        """Access data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._data[key].data

    def get_1d(self, key: str) -> np.ndarray:
        """Access 1D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._get(key)

    def get(self, key: str) -> np.ndarray:
        """Access 2D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._get(key)


class Reader:
    """Data Loader."""

    def __init__(
        self,
        filepath: Path,
    ) -> None:
        """Instantiate BathyLoader.

        Args:
            filepath: Path: Filepath to data.
        """
        if filepath.suffix == ".mat":
            self._core = MatReader(filepath=filepath)
        elif filepath.suffix == ".nc":
            self._core = NCReader(filepath=filepath)

    def get_1d(self, key: str) -> np.ndarray:
        """Access 1D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._core.get_1d(key=key)

    def get(self, key: str) -> np.ndarray:
        """Access 2D data element.

        Args:
            key (str): Data key.

        Returns:
            np.ndarray: Data value.
        """
        return self._core.get(key=key)
