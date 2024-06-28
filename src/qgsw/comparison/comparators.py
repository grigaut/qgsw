"""Comparison functions."""

from typing import Callable

import torch


class Comparator:
    """Comparator."""

    def __init__(self) -> None:
        """Instantiate Comparator."""

    def _raise_if_unmatching_size(
        self,
        top: torch.Tensor,
        bottom: torch.Tensor,
    ) -> None:
        if top.shape != bottom.shape:
            msg = "Top and Bottom tensor shapes don't match."
            raise ValueError(msg)

    def _compute_mask(
        self,
        top: torch.Tensor,
        bottom: torch.Tensor,
    ) -> torch.Tensor: ...

    def compare(self, top: torch.Tensor, bottom: torch.Tensor) -> Callable:
        """Compare.

        Args:
            top (torch.Tensor): _description_
            bottom (torch.Tensor): _description_

        Returns:
            Callable: _description_
        """
        self._raise_if_unmatching_size(top, bottom)
        mask = self._compute_mask(top, bottom)
        return self._compare(top.where(mask), bottom.where(mask))
