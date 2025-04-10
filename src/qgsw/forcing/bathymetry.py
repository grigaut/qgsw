"""Topography files loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import scipy.interpolate
import scipy.io
import scipy.ndimage
import skimage.morphology
import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.data.loaders import BathyLoader
from qgsw.exceptions import UnitError
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.configs.bathymetry import BathyConfig
    from qgsw.spatial.core.grid import Grid2D


class Bathymetry:
    """Bathymetry."""

    _required_xy_unit: Unit = Unit.DEG

    def __init__(
        self,
        loader: BathyLoader,
        interpolation_method: str,
        lake_min_area: float,
        island_min_area: float,
        htop_ocean: float,
    ) -> None:
        """Instantiate Bathymetry."""
        self._lake_min = lake_min_area
        self._island_min = island_min_area
        self._htop = htop_ocean
        self._lon, self._lat, self._bathy = loader.retrieve()
        self._interpolation = scipy.interpolate.RegularGridInterpolator(
            (self.lons, self.lats),
            self.elevation,
            method=interpolation_method,
        )

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

    def interpolate(
        self,
        grid_2d: Grid2D,
    ) -> np.ndarray:
        """Interpolate bathymetry on a given grid.

        Args:
            grid_2d (Grid2D): 2D grid.

        Returns:
            torch.Tensor: Interpolation of bathymetry on the given grid.
        """
        if grid_2d.xy_unit != self._required_xy_unit:
            msg = f"Grid2D xy unit must be {self._required_xy_unit}."
            raise UnitError(msg)
        return self._interpolation(grid_2d.xy)

    def compute_land_mask(
        self,
        grid_2d: Grid2D,
    ) -> torch.Tensor:
        """Compute land mask over a given grid.

        Args:
            grid_2d (Grid2D): 2D grid.

        Returns:
            torch.Tensor: Boolean mask with 1 over land cells and 0 elsewhere.
        """
        land = self.interpolate(grid_2d=grid_2d) > 0
        # remove small ocean inclusions in land
        land_without_lakes: np.ndarray = skimage.morphology.ds_closing(
            land,
            area_threshold=self._lake_min,
        )
        # remove small land inclusion in ocean
        land: np.ndarray = np.logical_not(
            skimage.morphology.ds_closing(
                np.logical_not(land_without_lakes),
                area_threshold=self._island_min,
            ),
        )
        return (
            torch.from_numpy(land).type(torch.float64).to(device=DEVICE.get())
        )

    def compute_ocean_mask(
        self,
        grid_2d: Grid2D,
    ) -> torch.Tensor:
        """Compute ocean mask over a given grid.

        Args:
            grid_2d (Grid2D): 2D grid.

        Returns:
            torch.Tensor: Boolean mask with 1 over ocean cells and 0 elsewhere.
        """
        interp_bathy = self.interpolate(grid_2d=grid_2d)
        ocean = interp_bathy < self._htop
        # Remove small land inclusions in ocean
        ocean_without_islands: np.ndarray = skimage.morphology.ds_closing(
            ocean,
            area_threshold=self._island_min,
        )
        # Remove small ocean inclusions in land
        ocean_without_lakes: np.ndarray = np.logical_not(
            skimage.morphology.ds_closing(
                np.logical_not(ocean_without_islands),
                area_threshold=self._lake_min,
            ),
        )
        ocean = self._remove_isolated_land(
            ocean_without_lakes.astype("float64"),
        )
        return (
            torch.from_numpy(ocean).type(torch.float64).to(device=DEVICE.get())
        )

    def _remove_isolated_land(self, ocean_mask: np.ndarray) -> np.ndarray:
        """Remove land cells surrounded by at least 3 ocean cells.

        Args:
            ocean_mask (np.ndarray): Ocean mask (1 over ocean cells else 0).

        Returns:
            np.ndarray: Corrected mask.
        """
        for _ in range(100):
            land_top = ocean_mask[:-2, 1:-1]
            land_below = ocean_mask[2:, 1:-1]
            land_left = ocean_mask[1:-1, :-2]
            land_right = ocean_mask[1:-1, 2:]
            # Number of ocean cells surrounding land
            nb_ocean_neigh = land_top + land_below + land_left + land_right
            has_3_neigh = nb_ocean_neigh > 2.5  # noqa: PLR2004
            # Land cells
            is_land = 1 - ocean_mask[1:-1, 1:-1]
            # Land cells with more than 3 ocean cells around
            ocean_mask[1:-1, 1:-1] += is_land * has_3_neigh
        return (ocean_mask > 0.5).astype("float64")  # noqa: PLR2004

    def compute_bottom_topography(
        self,
        grid_xy: tuple[torch.Tensor, torch.Tensor],
    ) -> np.ndarray:
        """Compute botoom topography.

        Args:
            grid_xy (tuple[torch.Tensor, torch.Tensor]): xy grid.

        Returns:
            np.ndarray: Bottom Topography.
        """
        bottom_topography = 4000 + np.clip(self.interpolate(grid_xy), -4000, 0)
        return np.clip(
            scipy.ndimage.gaussian_filter(bottom_topography, 3.0),
            0,
            150,
        )

    def compute_land_mask_w(self, grid_2d: Grid2D) -> torch.Tensor:
        """Pad land mask with land border and perform 2D average.

        Args:
            grid_2d (Grid2D): 2D grid.

        Returns:
            torch.Tensor: Land mask W.
        """
        land = self.compute_land_mask(grid_2d=grid_2d)
        unsqueezed = land.unsqueeze(0).unsqueeze(0)
        padded = F.pad(unsqueezed, (1, 1, 1, 1), value=1.0)
        avg_2d = F.avg_pool2d(padded, (2, 2), stride=(1, 1))[0, 0]
        return avg_2d > 0.5  # noqa: PLR2004

    @classmethod
    def from_config(cls, bathy_config: BathyConfig) -> Self:
        """Construct the Bathymetry given a BathyConfig object.

        Args:
            bathy_config (BathyConfig): Script Configuration Object.

        Returns:
            Self: Corresponding Bathymetry.
        """
        return cls(
            loader=BathyLoader.from_config(bathy_config),
            interpolation_method=bathy_config.interpolation_method,
            lake_min_area=bathy_config.lake_min_area,
            island_min_area=bathy_config.island_min_area,
            htop_ocean=bathy_config.htop_ocean,
        )
