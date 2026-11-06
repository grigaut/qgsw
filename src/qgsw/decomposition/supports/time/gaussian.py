"""Gaussian time supports."""

# ruff: noqa: D107, D102
from typing import Any

import torch
from torch._tensor import Tensor

from qgsw import specs
from qgsw.decomposition.supports.time.base import TimeSupportFunction


class GaussianTimeSupport(TimeSupportFunction):
    """Gaussian time support."""

    def __init__(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, Tensor],
    ) -> None:
        self._space = space_fields
        self.params = time_params
        self.unfreeze_normalization()

    __init__.__doc__ = TimeSupportFunction.__init__.__doc__

    def normalize(
        self, t: torch.Tensor, lvl: int, exp: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalization.

        Args:
            t (torch.Tensor): Time to normalize at.
            lvl (int): Lvl to normalize at.
            exp (torch.Tensor): Field.

        Returns:
            torch.Tensor: Normalized field.
        """
        denom = exp.sum(dim=-1, keepdim=True)
        if self.norms:
            frozen = (t >= self.freeze_t).unsqueeze(-1)
            denom = torch.where(frozen, self.norms[lvl], denom)
        return exp / denom

    def freeze_normalization(self, t: torch.Tensor) -> None:
        """Freeze normalization after a given time.

        Args:
            t (torch.Tensor): Time to freeze normalizing at.
        """
        self.norms = {}
        self.freeze_t = t.clone()
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            self.norms[lvl] = torch.exp(-((t - tc) ** 2) / (st) ** 2).sum(
                dim=-1
            )

    def unfreeze_normalization(self) -> None:
        """Unfreeze normalization."""
        self.norms = {}

    def retrieve_support(self, lvl: int, t: torch.Tensor) -> torch.Tensor:
        """Retrieve time support at given level.

        Args:
            lvl (int): Level.
            t (torch.Tensor): T locations.

        Returns:
            torch.Tensor: Time support.
        """
        centers = self.params[lvl]["centers"]
        st: float = self.params[lvl]["sigma_t"]
        tc = torch.tensor(centers, **specs.from_tensor(t))
        exp = torch.exp(-((t[..., None] - tc) ** 2) / (st) ** 2)
        return self.normalize(t, lvl, exp)

    def retrieve_supports(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Retrieve time supports.

        Args:
            t (torch.Tensor): T locations.

        Returns:
            torch.Tensor: Time supports mapping.
        """
        return {lvl: self.retrieve_support(lvl, t) for lvl in self.params}

    def retrieve_dt_support(self, lvl: int, t: torch.Tensor) -> torch.Tensor:
        """Retrieve dt time support at given level.

        Args:
            lvl (int): Level.
            t (torch.Tensor): T locations.

        Returns:
            torch.Tensor: dt time support.
        """
        centers = self.params[lvl]["centers"]
        st: float = self.params[lvl]["sigma_t"]
        tc = torch.tensor(centers, **specs.from_tensor(t))
        exp = torch.exp(-((t[..., None] - tc) ** 2) / (st) ** 2)
        dt_exp = -2 * (t[..., None] - tc) / st**2 * exp
        return self.dt_normalized(t, lvl, exp, dt_exp)

    def retrieve_dt_supports(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Retrieve dt time supports.

        Args:
            t (torch.Tensor): T locations.

        Returns:
            torch.Tensor: dt time supports mapping.
        """
        return {lvl: self.retrieve_dt_support(lvl, t) for lvl in self.params}

    def decompose(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        for lvl in self.params:
            e = self.retrieve_support(lvl, t)
            fields[lvl] = torch.einsum("...t,txy->...xy", e, self._space[lvl])
        return fields

    decompose.__doc__ = TimeSupportFunction.decompose.__doc__

    def __call__(self, t: Tensor) -> Tensor:
        fields = self.decompose(t)
        return sum(fields.values()) / len(self.params)

    __call__.__doc__ = TimeSupportFunction.__call__.__doc__

    def dt_normalized(
        self,
        t: torch.Tensor,
        lvl: int,
        exp: torch.Tensor,
        dt_exp: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize time derivative of the field.

        Args:
            t (torch.Tensor): Time to normalize at.
            lvl (int): Lvl to normalize at.
            exp (torch.Tensor): Field.
            dt_exp (torch.Tensor): Time derivative of the field.

        Returns:
            torch.Tensor: Time derivative of normalized field.
        """
        exp_s = exp.sum(dim=-1, keepdim=True)
        if self.norms:
            frozen = (t >= self.freeze_t).unsqueeze(-1)
            exp_s = torch.where(frozen, torch.ones_like(exp_s), exp_s)
        softmax_grad = (
            dt_exp * exp_s - exp * dt_exp.sum(dim=-1, keepdim=True)
        ) / exp_s**2
        if not self.norms:
            return softmax_grad
        return torch.where(frozen, dt_exp / self.norms[lvl], softmax_grad)

    def decompose_dt(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        for lvl in self.params:
            dt_e = self.retrieve_dt_support(lvl, t)
            fields[lvl] = torch.einsum(
                "...t,txy->...xy", dt_e, self._space[lvl]
            )
        return fields

    decompose_dt.__doc__ = TimeSupportFunction.decompose_dt.__doc__

    def dt(self, t: Tensor) -> Tensor:
        fields = self.decompose_dt(t)
        return sum(fields.values()) / len(self.params)

    dt.__doc__ = TimeSupportFunction.dt.__doc__
