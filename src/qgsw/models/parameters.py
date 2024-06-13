"""Model Parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.masks import Masks
from qgsw.models.exceptions import InvalidModelParameterError

if TYPE_CHECKING:
    from qgsw.physics.coriolis.beta_plane import BetaPlane
    from qgsw.spatial.core.discretization import SpaceDiscretization3D


class ModelParamChecker:
    """Model Parameters."""

    dtype: torch.dtype
    device: str
    _n_ens: int = 1
    _masks: Masks = None
    _dt: float
    _slip_coef: float = 0.0
    _bottom_drag: float = 0.0
    _taux: torch.Tensor | float
    _tauy: torch.Tensor | float

    def __init__(
        self,
        *,
        space_3d: SpaceDiscretization3D,
        g_prime: torch.Tensor,
        beta_plane: BetaPlane,
    ) -> None:
        """Model Instantiation.

        Args:
            space_3d (SpaceDiscretization3D): Space Discretization
            g_prime (torch.Tensor): Reduced Gravity Values Tensor.
            beta_plane (BetaPlane): Beta Plane.
        """
        # Set up
        verbose.display(
            msg=f"dtype: {self.dtype}.",
            trigger_level=2,
        )
        verbose.display(
            msg=f"device: {self.device}",
            trigger_level=2,
        )
        ## Space
        self._space = space_3d
        self._beta_plane = beta_plane
        # h
        self._set_H(space_3d.h.xyh.h)
        ## gravity
        self._set_g_prime(g_prime)

    @property
    def space(self) -> SpaceDiscretization3D:
        """3D Space Discretization."""
        return self._space

    @property
    def dt(self) -> float:
        """Timestep value."""
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._set_dt(dt)

    @property
    def slip_coef(self) -> float:
        """Slip coefficient."""
        return self._slip_coef

    @slip_coef.setter
    def slip_coef(self, slip_coefficient: float) -> None:
        self._set_slip_coef(slip_coefficient)

    @property
    def bottom_drag_coef(self) -> float:
        """Bottom drag coefficient."""
        return self._bottom_drag

    @bottom_drag_coef.setter
    def bottom_drag_coef(self, bottom_drag: float) -> None:
        self._set_bottom_drag(bottom_drag)

    @property
    def n_ens(self) -> int:
        """Number of ensembles."""
        return self._n_ens

    @n_ens.setter
    def n_ens(self, n_ens: int) -> None:
        self._set_n_ens(n_ens)

    @property
    def masks(self) -> Masks:
        """Masks."""
        if self._masks is None:
            mask = torch.ones(
                self.space.nx,
                self.space.ny,
                dtype=self.dtype,
                device=self.device,
            )
            self._masks = Masks(mask)
        return self._masks

    @masks.setter
    def masks(self, mask: torch.Tensor) -> None:
        self._set_masks(mask)

    @property
    def beta_plane(self) -> BetaPlane:
        """Beta Plane parmaters."""
        return self._beta_plane

    @property
    def H(self) -> torch.Tensor:  # noqa: N802
        """Layers thickness."""
        return self._H

    @property
    def g_prime(self) -> torch.Tensor:
        """Reduced Gravity."""
        return self._g_prime

    @property
    def g(self) -> float:
        """Reduced gravity in top layer."""
        return self.g_prime[0]

    @property
    def taux(self) -> torch.Tensor | float:
        """Tau x."""
        return self._taux

    @property
    def tauy(self) -> torch.Tensor | float:
        """Tau y."""
        return self._tauy

    def _set_dt(self, dt: float) -> None:
        """TimeStep Setter.

        Args:
            dt (float): Timestep (s)

        Raises:
            InvalidModelParameterError: If the timestep is negative.
        """
        if dt <= 0:
            msg = "Timestep must be greater than 0."
            raise InvalidModelParameterError(msg)
        self._dt = dt
        verbose.display(f"dt set to {self.dt}", trigger_level=1)

    def _set_slip_coef(self, slip_coefficient: float) -> None:
        """Set the slip coefficient.

        Args:
            slip_coefficient (float): Slip coefficient.

        Raises:
            InvalidModelParameterError: If the lsip coef is not in [0,1]
        """
        # Verify value in [0,1]
        if (slip_coefficient < 0) or (slip_coefficient > 1):
            msg = f"slip coefficient must be in [0, 1], got {slip_coefficient}"
            raise InvalidModelParameterError(msg)
        self._clip_coef = slip_coefficient

    def _set_bottom_drag(self, bottom_drag: float) -> None:
        """Set th ebottom drag coefficient.

        Args:
            bottom_drag (float): Bottom drag coefficient.
        """
        self._bottom_drag = bottom_drag

    def _set_n_ens(self, n_ens: int) -> None:
        """Set the number of ensembles.

        Args:
            n_ens (int): Number of ensembles.

        Raises:
            InvalidModelParameterError: If the number of instance is not an int
            InvalidModelParameterError: If the number of ensemble is negative.
        """
        if not isinstance(n_ens, int):
            msg = "n_ens must be an integer."
            raise InvalidModelParameterError(msg)
        if n_ens <= 0:
            msg = "n_ens must be greater than 1."
            raise InvalidModelParameterError(msg)
        self._n_ens = n_ens

    def _set_masks(self, mask: torch.Tensor) -> None:
        """Set the masks.

        Args:
            mask (torch.Tensor): Mask tensor.

        Raises:
            InvalidModelParameterError: If the mask has incorrect shape.
            InvalidModelParameterError: If the mask is not inly 1s and 0s.
        """
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
                f"{'Non-trivial' if len(vals)==2 else 'Trivial'}"  # noqa: PLR2004
                " mask provided"
            ),
            trigger_level=2,
        )
        self._masks = Masks(mask)

    def _set_H(  # noqa: N802
        self,
        h: torch.Tensor,
    ) -> None:
        """Validate H (unperturbed layer thickness) input value.

        Args:
            h (torch.Tensor): Layers Thickness.
        """
        if len(h.shape) < 3:  # noqa: PLR2004
            msg = (
                "H must be a nz x ny x nx tensor "
                "with nx=1 or ny=1 if H does not vary "
                f"in x or y direction, got shape {h.shape}."
            )
            raise InvalidModelParameterError(msg)
        self._H = h

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
        if is_tensorx and (taux.shape != (self.space.nx - 1, self.space.ny)):
            msg = (
                "Tau_x Tensor must be "
                f"{(self.space.nx-1, self.space.ny)}-shaped."
            )
            raise InvalidModelParameterError(msg)
        self._taux = taux

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
        if is_tensory and (tauy.shape != (self.space.nx, self.space.ny - 1)):
            msg = (
                "Tau_y Tensor must be "
                f"{(self.space.nx, self.space.ny-1)}-shaped."
            )
            raise InvalidModelParameterError(msg)
        self._tauy = tauy
