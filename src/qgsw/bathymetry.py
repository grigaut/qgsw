"""Topography files loaders."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import scipy.io
import skimage.morphology
import xarray
from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np
    import torch

    from qgsw.configs import BathyConfig, RunConfig


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


class LandBathyFilter:
    """Land Bathymetry filter."""

    def __init__(
        self,
        lake_min_area: int,
        island_min_area: int,
    ) -> None:
        """Instantiate LandBathyFilter.

        Args:
            lake_min_area (int): Lake minimum area.
            island_min_area (int): Island minimum area.
        """
        self._lake = lake_min_area
        self._island = island_min_area

    def filter_sign(self, bathy_field: torch.Tensor) -> torch.Tensor:
        """Filter positive or negative bathymetry.

        Args:
            bathy_field (torch.Tensor): Bathymetry field.

        Returns:
            torch.Tensor: Filtered bathymetry, 1 for positive bathymetry else 0
        """
        return bathy_field > 0

    def filter_small_lakes(self, bathy_sign: torch.Tensor) -> torch.Tensor:
        """Remove small lakes.

        Args:
            bathy_sign (torch.Tensor): Binary (land / Ocean) bathymetry.

        Returns:
            torch.Tensor: Filtered Bathymetry.
        """
        return skimage.morphology.area_closing(
            bathy_sign, area_threshold=self._lake
        )

    def filter_small_islands(self, bathy_sign: torch.Tensor) -> torch.Tensor:
        """Remove small islands.

        Args:
            bathy_sign (torch.Tensor): Binary (land / Ocean) bathymetry.

        Returns:
            torch.Tensor: Filtered Bathymetry.
        """
        return ~(
            skimage.morphology.area_closing(
                ~bathy_sign,
                area_threshold=self._island,
            )
        )


class OceanBathyFilter:
    """Ocean bathymetry Filter."""


class Bathymetry:
    """Bathymetry."""

    def __init__(self, bathy_config: BathyConfig) -> None:
        """Instantiate Bathymetry."""
        self._config = bathy_config
        loader = BathyLoader.from_url(
            url=self._config.url,
            savefolder=self._config.folder,
        )
        self._lon, self._lat, self._bathy = loader.retrieve_bathy()

    @property
    def lons(self) -> np.ndarray:
        """Bathymetry longitude array."""
        return self._lon

    @property
    def lats(self) -> np.ndarray:
        """Bathymetry latitude array."""
        return self._lat

    @property
    def elevation(self) -> np.ndarray:
        """Bahymetry."""
        return self._bathy

    @classmethod
    def from_runconfig(cls, run_config: RunConfig) -> Self:
        """Construct the Bathymetry given a RunConfig object.

        Args:
            run_config (RunConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Bathymetry.
        """
        return cls(bathy_config=run_config.bathy)
