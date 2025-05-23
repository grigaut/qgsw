"""Model Parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.exceptions import (
    InvalidModelParameterError,
    UnsetTimestepError,
)
from qgsw.masks import Masks
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.grid_conversion import points_to_surfaces
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.spatial.core.discretization import (
        SpaceDiscretization2D,
        SpaceDiscretization3D,
    )
    from qgsw.specs._utils import Device


class ModelParamChecker:
    """Model Parameters."""

    dtype: torch.dtype
    device: Device
    _n_ens: int = 1
    _masks: Masks
    _dt: float
    _slip_coef: float = 0.0
    _bottom_drag: float = 0.0
    _taux: torch.Tensor | float
    _tauy: torch.Tensor | float

    def __init__(
        self,
        *,
        space_2d: SpaceDiscretization2D,
        H: torch.Tensor,  # noqa: N803
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
    ) -> None:
        """Model Instantiation.

        Args:
            space_2d (SpaceDiscretization2D): Space Discretization
            H (torch.tensor): reference layer depth
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (Beta_Plane): Beta plane.
        """
        # Set up
        verbose.display(
            msg=f"dtype: {self.dtype}.",
            trigger_level=2,
        )
        verbose.display(
            msg=f"device: {self.device.get()}",
            trigger_level=2,
        )
        ## Space
        self._space = space_2d.add_h(Coordinates1D(points=H, unit=Unit.M))
        # Beta-plane
        self._set_beta_plane(beta_plane)
        # h
        self._set_H(self._space.h.xyh.h)
        ## gravity
        self._set_g_prime(g_prime.unsqueeze(1).unsqueeze(1))

    @property
    def space(self) -> SpaceDiscretization3D:
        """3D Space Discretization."""
        return self._space

    @property
    def dt(self) -> float:
        """Timestep value."""
        try:
            return self._dt
        except AttributeError as e:
            raise UnsetTimestepError from e

    @dt.setter
    def dt(self, dt: float) -> None:
        if dt <= 0:
            msg = "Timestep must be greater than 0."
            raise InvalidModelParameterError(msg)
        self._dt = dt
        verbose.display(
            f"{self.__class__.__name__}: dt set to {self.dt}",
            trigger_level=1,
        )

    @property
    def slip_coef(self) -> float:
        """Slip coefficient."""
        return self._slip_coef

    @slip_coef.setter
    def slip_coef(self, slip_coefficient: float) -> None:
        # Verify value in [0,1]
        if (slip_coefficient < 0) or (slip_coefficient > 1):
            msg = f"slip coefficient must be in [0, 1], got {slip_coefficient}"
            raise InvalidModelParameterError(msg)
        self._slip_coef = slip_coefficient
        name = self.__class__.__name__
        verbose.display(
            f"{name}: Slip coefficient set to {slip_coefficient}",
            trigger_level=2,
        )

    @property
    def bottom_drag_coef(self) -> float:
        """Bottom drag coefficient."""
        return self._bottom_drag

    @bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag: float) -> None:
        self._bottom_drag = bottom_drag
        name = self.__class__.__name__
        verbose.display(
            f"{name}: Bottom drag set to {bottom_drag}",
            trigger_level=2,
        )

    @property
    def n_ens(self) -> int:
        """Number of ensembles."""
        return self._n_ens

    @n_ens.setter
    def n_ens(self, n_ens: int) -> None:
        if not isinstance(n_ens, int):
            msg = "n_ens must be an integer."
            raise InvalidModelParameterError(msg)
        if n_ens <= 0:
            msg = "n_ens must be greater than 1."
            raise InvalidModelParameterError(msg)
        self._n_ens = n_ens
        self._size_ens.set_to(self._n_ens)
        name = self.__class__.__name__
        verbose.display(
            f"{name}: Number of ensembles set to {n_ens}",
            trigger_level=2,
        )

    @property
    def masks(self) -> Masks:
        """Masks."""
        try:
            return self._masks
        except AttributeError:
            self._masks = Masks.empty(
                self.space.nx,
                self.space.ny,
                device=DEVICE.get(),
            )
            return self._masks

    @masks.setter
    def masks(self, mask: torch.Tensor) -> None:
        shape = mask.shape[0], mask.shape[1]
        # Verify shape
        if shape != (self.space.nx, self.space.ny):
            msg = (
                "Invalid mask shape "
                f"{shape}!=({self.space.nx},{self.space.ny})"
            )
            raise InvalidModelParameterError(msg)
        vals = torch.unique(mask).tolist()
        # Verify mask values
        if not all(v in [0, 1] for v in vals) or vals == [0]:
            msg = f"Invalid mask with non-binary values : {vals}"
            raise InvalidModelParameterError(msg)
        verbose.display(
            msg=(
                f"{'Non-trivial' if len(vals) == 2 else 'Trivial'}"  # noqa: PLR2004
                " mask provided"
            ),
            trigger_level=2,
        )
        self._masks = Masks(mask)

    @property
    def beta_plane(self) -> BetaPlane:
        """Beta Plane paramters."""
        return self._beta_plane

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness.

        └── (nl, 1, 1)-shaped.
        """
        return self._H

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity.

        └── (nl, 1, 1)-shaped
        """
        return self._g_prime

    @property
    def g(self) -> float:
        """Reduced gravity in top layer.

        (n_ens,)-shaped.
        """
        return self.g_prime[0, 0, 0]

    @property
    def taux(self) -> torch.Tensor | float:
        """Tau x."""
        return self._taux

    @property
    def tauy(self) -> torch.Tensor | float:
        """Tau y."""
        return self._tauy

    def _set_beta_plane(self, beta_plane: BetaPlane) -> None:
        if not isinstance(beta_plane, BetaPlane):
            msg = "beta_plane should be of type `BetaPlane`."
            raise TypeError(msg)
        verbose.display(
            f"{self.__class__.__name__}: Beta-plane set to {beta_plane}",
            trigger_level=2,
        )
        self._beta_plane = beta_plane

    def _set_H(  # noqa: N802
        self,
        h: torch.Tensor,
    ) -> None:
        """Validate H (unperturbed layer thickness) input value.

        Args:
            h (torch.Tensor): Layers Thickness.
                └── (nl, 1, 1)-shaped
        """
        if len(h.shape) < 3:  # noqa: PLR2004
            msg = (
                "H must be a nz x ny x nx tensor "
                "with nx=1 or ny=1 if H does not vary "
                f"in x or y direction, got shape {h.shape}."
            )
            raise InvalidModelParameterError(msg)
        self._H = h
        name = self.__class__.__name__
        verbose.display(
            f"{name}: H set to {h}",
            trigger_level=2,
        )

    def _set_g_prime(self, g_prime: torch.Tensor) -> None:
        """Set g_prime values.

        Args:
            g_prime (torch.Tensor): g_prime.
        """
        if g_prime.shape != self._H.shape:
            msg = (
                f"Inconsistent shapes for g_prime ({g_prime.shape}) "
                f"and H ({self._H.shape})"
            )
            raise InvalidModelParameterError(msg)
        self._g_prime = g_prime
        name = self.__class__.__name__
        verbose.display(
            f"{name}: g' set to {g_prime}",
            trigger_level=2,
        )

    def _set_taux(self, taux: torch.Tensor | float) -> None:
        """Set taux value.

        Args:
            taux (torch.Tensor | float): Tau x value.

        Raises:
            InvalidModelParameterError: If taux has invalid type.
            InvalidModelParameterError: If taux has invalid shape.
        """
        is_tensorx = isinstance(taux, torch.Tensor)
        if (not isinstance(taux, float)) and (not is_tensorx):
            msg = "taux must be a float or a Tensor"
            raise InvalidModelParameterError(msg)
        if is_tensorx and (taux.shape != (self.space.nx, self.space.ny + 1)):
            msg = (
                "Tau_x Tensor must be "
                f"{(self.space.nx, self.space.ny + 1)}-shaped."
            )
            raise InvalidModelParameterError(msg)
        self._taux = points_to_surfaces(taux)

    def _set_tauy(self, tauy: torch.Tensor | float) -> None:
        """Set tauy value.

        Args:
            tauy (torch.Tensor | float): Tau x value.

        Raises:
            InvalidModelParameterError: If tauy has invalid type.
            InvalidModelParameterError: If tauy has invalid shape.
        """
        is_tensory = isinstance(tauy, torch.Tensor)
        if (not isinstance(tauy, float)) and (not is_tensory):
            msg = "tauy must be a float or a Tensor"
            raise InvalidModelParameterError(msg)
        if is_tensory and (tauy.shape != (self.space.nx + 1, self.space.ny)):
            msg = (
                "Tau_y Tensor must be "
                f"{(self.space.nx + 1, self.space.ny)}-shaped."
            )
            raise InvalidModelParameterError(msg)
        self._tauy = points_to_surfaces(tauy)
