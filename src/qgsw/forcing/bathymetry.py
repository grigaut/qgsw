"""Topography files loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import scipy.io
import scipy.ndimage
import skimage.morphology
import torch
import torch.nn.functional as F  # noqa: N812
from typing_extensions import Self

from qgsw.data.loaders import BathyLoader
from qgsw.spatial.units._units import DEGREES, Unit
from qgsw.spatial.units.exceptions import UnitError
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.bathymetry import BathyConfig
    from qgsw.mesh.mesh import Mesh2D


class Bathymetry:
    """Bathymetry."""

    _required_xy_unit: Unit = DEGREES

    def __init__(self, bathy_config: BathyConfig) -> None:
        """Instantiate Bathymetry."""
        self._config = bathy_config
        loader = BathyLoader(config=self._config.data)
        self._lon, self._lat, self._bathy = loader.retrieve()
        self._interpolation = scipy.interpolate.RegularGridInterpolator(
            (self.lons, self.lats),
            self.elevation,
            method=self._config.interpolation_method,
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
        mesh_2d: Mesh2D,
    ) -> np.ndarray:
        """Interpolate bathymetry on a given mesh.

        Args:
            mesh_2d (Mesh2D): 2D mesh.

        Returns:
            torch.Tensor: Interpolation of bathymetry on the given mesh.
        """
        if mesh_2d.xy_unit != self._required_xy_unit:
            msg = f"Mesh2D xy unit must be {self._required_xy_unit}."
            raise UnitError(msg)
        return self._interpolation(mesh_2d.xy)

    def compute_land_mask(
        self,
        mesh_2d: Mesh2D,
    ) -> torch.Tensor:
        """Compute land mask over a given mesh.

        Args:
            mesh_2d (Mesh2D): 2D mesh.

        Returns:
            torch.Tensor: Boolean mask with 1 over land cells and 0 elsewhere.
        """
        land = self.interpolate(mesh_2d=mesh_2d) > 0
        # remove small ocean inclusions in land
        land_without_lakes: np.ndarray = skimage.morphology.area_closing(
            land,
            area_threshold=self._config.lake_min_area,
        )
        # remove small land inclusion in ocean
        land: np.ndarray = np.logical_not(
            skimage.morphology.area_closing(
                np.logical_not(land_without_lakes),
                area_threshold=self._config.island_min_area,
            ),
        )
        return torch.from_numpy(land).type(torch.float64).to(device=DEVICE)

    def compute_ocean_mask(
        self,
        mesh_2d: Mesh2D,
    ) -> torch.Tensor:
        """Compute ocean mask over a given mesh.

        Args:
            mesh_2d (Mesh2D): 2D mesh.

        Returns:
            torch.Tensor: Boolean mask with 1 over ocean cells and 0 elsewhere.
        """
        interp_bathy = self.interpolate(mesh_2d=mesh_2d)
        ocean = interp_bathy < self._config.htop_ocean
        # Remove small land inclusions in ocean
        ocean_without_islands: np.ndarray = skimage.morphology.area_closing(
            ocean,
            area_threshold=self._config.island_min_area,
        )
        # Remove small ocean inclusions in land
        ocean_without_lakes: np.ndarray = np.logical_not(
            skimage.morphology.area_closing(
                np.logical_not(ocean_without_islands),
                area_threshold=self._config.lake_min_area,
            ),
        )
        ocean = self._remove_isolated_land(
            ocean_without_lakes.astype("float64"),
        )
        return torch.from_numpy(ocean).type(torch.float64).to(device=DEVICE)

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
        mesh_xy: tuple[torch.Tensor, torch.Tensor],
    ) -> np.ndarray:
        """Compute botoom topography.

        Args:
            mesh_xy (tuple[torch.Tensor, torch.Tensor]): xy mesh.

        Returns:
            np.ndarray: Bottom Topography.
        """
        bottom_topography = 4000 + np.clip(self.interpolate(mesh_xy), -4000, 0)
        return np.clip(
            scipy.ndimage.gaussian_filter(bottom_topography, 3.0),
            0,
            150,
        )

    def compute_land_mask_w(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Pad land mask with land border and perform 2D average.

        Args:
            mesh_2d (Mesh2D): 2D mesh.

        Returns:
            torch.Tensor: Land mask W.
        """
        land = self.compute_land_mask(mesh_2d=mesh_2d)
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
        return cls(bathy_config=bathy_config)