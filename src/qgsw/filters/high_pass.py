"""High-Pass filters."""

from torch import Tensor

from qgsw.filters.gaussian import GaussianFilter1D, GaussianFilter2D


class GaussianHighPass1D(GaussianFilter1D):
    """1D Gaussian High-Pass filter."""

    def __call__(self, to_filter: Tensor) -> Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        low_pass = super().__call__(to_filter)
        return to_filter - low_pass


class GaussianHighPass2D(GaussianFilter2D):
    """2D Gaussian High-Pass filter."""

    def __call__(self, to_filter: Tensor) -> Tensor:
        """Perform filtering.

        Args:
            to_filter (torch.Tensor): Tensor to filter, (p,q)-shaped.

        Returns:
            torch.Tensor: Filtered tensor.
        """
        low_pass = super().__call__(to_filter)
        return to_filter - low_pass
