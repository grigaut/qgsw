"""Base classes for space-time decompositions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.specs import defaults

if TYPE_CHECKING:
    import torch

    from qgsw.decomposition.coefficients import DecompositionCoefs
TimeSupport = TypeVar("TimeSupport", bound=TimeSupportFunction)
SpaceSupport = TypeVar("SpaceSupport", bound=SpaceSupportFunction)


class SpaceTimeDecomposition(ABC, Generic[SpaceSupport, TimeSupport]):
    """Space-time decomposition."""

    @cached_property
    def order(self) -> int:
        """Decomposition order."""
        return len(self._space.keys())

    def __init__(
        self,
        space_params: dict[int, dict[str, Any]],
        time_params: dict[int, dict[str, Any]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the decomposition.

        Args:
            space_params (dict[int, dict[str, Any]]): Space parameters.
            time_params (dict[int, dict[str, Any]]): Time parameters.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Ddevice.
                Defaults to None.
        """
        self._check_validity(space_params, time_params)
        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params

    def _check_validity(
        self,
        space_params: dict[str, Any],
        time_params: dict[str, Any],
    ) -> None:
        """Check parameters validity."""
        if space_params.keys() != time_params.keys():
            msg = "Mismatching keys between space and time parameters."
            raise ValueError(msg)

    @abstractmethod
    def numel(self) -> int:
        """Total number of elements."""

    @abstractmethod
    def generate_random_coefs(self) -> DecompositionCoefs:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            DecompositionCoefs: Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """

    def set_coefs(self, coefs: DecompositionCoefs) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (DecompositionCoefs): Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """
        self._coefs = coefs

    @abstractmethod
    def localize(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """

    def localize_dx(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize x-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement x-derivative."
        raise NotImplementedError(msg)

    def localize_dy(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize y-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement y-derivative."
        raise NotImplementedError(msg)

    def localize_dx2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize xx-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement xx-derivative."
        raise NotImplementedError(msg)

    def localize_dy2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize yy-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement yy-derivative."
        raise NotImplementedError(msg)

    def localize_dx3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize xxx-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement xxx-derivative."
        raise NotImplementedError(msg)

    def localize_dy3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize yyy-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement yyy-derivative."
        raise NotImplementedError(msg)

    def localize_dydx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        """Localize xxy-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement xxy-derivative."
        raise NotImplementedError(msg)

    def localize_dxdy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        """Localize yyx-derivative of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement yyx-derivative."
        raise NotImplementedError(msg)

    def localize_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        """Localize laplacian of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement laplacian."
        raise NotImplementedError(msg)

    def localize_dx_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        """Localize x-derivative of laplacian of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement dx laplacian."
        raise NotImplementedError(msg)

    def localize_dy_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        """Localize y-derivative of laplacian of space support function.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        msg = "This decomposition does not implement dy laplacian."
        raise NotImplementedError(msg)
