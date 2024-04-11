"""Topography files loaders."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import scipy.io
import xarray
from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np


class BathyLoadingError(Exception):
    """Error occuring when loading bathymetry."""


class BathyLoader:
    """Bathymetry loader."""

    def __init__(
        self,
        filepath: Path,
    ) -> None:
        """Instantiate BathyLoader.

        Args:
            filepath: Path: Filepath to data.
        """
        self._filepath = filepath

    def retrieve_bathy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retrieve Bathymetry.

        Raises:
            BathyLoadingError: If the file has unsupported type.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            Longitude, Latitude, Bathymetry
        """
        if self._filepath.suffix == ".mat":
            return self._load_matfile()
        if self._filepath.suffix == ".nc":
            return self._load_netcdf()
        msg = "Unsupported file type."
        raise BathyLoadingError(msg)

    def _load_matfile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from MATLAB file.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            Longitude, Latitude, Bathymetry
        """
        data = scipy.io.loadmat(self._filepath)
        lon_bath: np.ndarray = data["lon_bathy"][:, 0]
        lat_bath: np.ndarray = data["lat_bathy"][:, 0]
        bathy: np.ndarray = data["bathy"].T
        return lon_bath, lat_bath, bathy

    def _load_netcdf(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from NetCDF file.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            Longitude, Latitude, Bathymetry
        """
        data = xarray.open_dataset(self._filepath)
        lon_bath = data["lon"].data
        lat_bath = data["lat"].data
        bathy = data["elevation"].data.T
        return lon_bath, lat_bath, bathy

    @classmethod
    def from_url(cls, url: str, savefolder: Path) -> Self:
        """Create BathyLoader from a URL.

        Args:
            url (str): URL to download dat from.
            savefolder (Path): Folder to save data in.

        Returns:
            Self: Instantiated BathyLoader.
        """
        savepath = savefolder.joinpath(Path(url).name)
        if not savepath.is_file():
            print(f"Downloading topo file {savepath} from {url}...")
            urllib.request.urlretrieve(url, savepath)
            print("..done")
        return cls(filepath=savepath)
