"""Utils."""

from collections.abc import Callable
from itertools import chain
from typing import Generic, Self, TypeVar

import torch

from qgsw import logging
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.utils.tensor_dict import iterate_over_dict

logger = logging.getLogger(__name__)


class EarlyStop:
    """Early stop."""

    def __init__(self, *, eps: float = 1e-5, stop_after: int = 10) -> None:
        """INstantiate EarlyStop.

        Args:
            eps (float, optional): Precision. Defaults to 1e-5.
            stop_after (int, optional): MAx allowed iterations on a plateau.
                Defaults to 10.
        """
        self.eps = eps
        self.stop_after = stop_after
        self.counter = 0
        self.previous_loss = None

    def step(self, loss: torch.Tensor) -> bool:
        """Check against loss value.

        Args:
            loss (torch.Tensor): Loss value.

        Returns:
            bool: True if loss has been stable.
        """
        if self.previous_loss is None:
            self.previous_loss = loss.detach()
            return False
        loss_ = self.previous_loss
        if ((loss - loss_).abs() / (loss_.abs() + 1e-12)) < self.eps:
            self.counter += 1
        else:
            self.counter = 0
            self.previous_loss = loss.detach()
        return self.counter >= self.stop_after


T = TypeVar("T", bound=torch.Tensor | DecompositionCoefs)


class Param(Generic[T]):
    """Param."""

    def __init__(
        self,
        data: T,
        lr: float = 1,
    ) -> None:
        """Instantiate Param.

        Args:
            data (T): Data.
            scale_fn (Callable[[T], T] | None, optional): Sacling function.
                Defaults to None.
            lr (float, optional): Learning rate.. Defaults to 1.
        """
        self.data = data
        self.lr = lr
        self.with_scaling(None)

    @property
    def optim(self) -> bool:
        """Whether to optim."""
        return self.data.requires_grad

    @property
    def optim_list(self) -> list[torch.Tensor]:
        """Opitm list."""
        if self.optim:
            if isinstance(self.data, DecompositionCoefs):
                return list(self.data.values())
            return [self.data]
        return []

    @property
    def optim_numel(self) -> int:
        """Numel."""
        if self.optim:
            return self.data.numel()
        return 0

    @property
    def value(self) -> T:
        """Scaled value."""
        return self.scale_fn(self.data * self.lr)

    def with_scaling(self, fn: Callable[[T], T]) -> Self:
        """Change scaling function."""
        if fn is None:
            fn: Callable[[T], T] = lambda t: t
        self.scale_fn = fn
        return self


class TrackParams:
    """Track params."""

    def __init__(
        self,
        coefs: dict[str, Param[DecompositionCoefs]],
        tensors: dict[str, Param[torch.Tensor]],
    ) -> None:
        """Instantiate param tracker.

        Args:
            coefs (dict[str, Param[DecompositionCoefs]]): Coefs.
            tensors (dict[str, Param[torch.Tensor]]): Tensors.
        """
        self.coefs = coefs
        self.tensors = tensors
        self.retrieve_best_data_from_current()
        self.best_loss = torch.tensor(float("inf"))

    def _detach_clone_tensors(
        self, params: dict[str, Param[torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Detach clone tensors."""
        return iterate_over_dict(
            {k: p.data for k, p in params.items()},
            lambda t: t.detach().clone(),
        )

    def _detach_clone_coefs(
        self, params: dict[str, Param[DecompositionCoefs]]
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Detach and clone coefs."""
        return iterate_over_dict(
            {k: p.data.to_dict() for k, p in params.items()},
            lambda t: t.detach().clone(),
        )

    def retrieve_best_data_from_current(self) -> None:
        """Retrieve best data using current data states."""
        self.best_coefs_data = self._detach_clone_coefs(self.coefs)
        self.best_tensors_data = self._detach_clone_tensors(self.tensors)

    def step(self, loss: torch.Tensor) -> None:
        """Make a step to evaluate loss."""
        if torch.isnan(loss) or loss.item() > self.best_loss.item():
            return
        self.retrieve_best_data_from_current()
        self.best_loss = torch.clone(loss.detach())

    def restore_best(self) -> None:
        """Restore best parameters."""
        for k, p in self.coefs.items():
            best = self.best_coefs_data[k]
            for i in range(p.data.order):
                p.data[i].data.copy_(best[i].data)
        for k, p in self.tensors.items():
            best = self.best_tensors_data[k]
            p.data.copy_(best.data)

    def get_optim_list(self) -> list[torch.Tensor]:
        """Retrieve optimisable param list."""
        return list(
            chain.from_iterable(
                obj.optim_list
                for container in (self.coefs.values(), self.tensors.values())
                for obj in container
            )
        )

    def get_optim_numel(self) -> int:
        """Retrieve numel for optim params."""
        return sum(p.optim_numel for p in self.coefs.values()) + sum(
            p.optim_numel for p in self.tensors.values()
        )

    def retrieve_scaled(
        self,
    ) -> tuple[dict[str, DecompositionCoefs], dict[str, torch.Tensor]]:
        """Retrieve scaled params."""
        return (
            {k: p.value for k, p in self.coefs.items()},
            {k: p.value for k, p in self.tensors.items()},
        )

    def retrieve_best_data_scaled(
        self,
    ) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, torch.Tensor]]:
        """Retrieve best data scaled."""
        return (
            {
                k: self.coefs[k]
                .scale_fn(DecompositionCoefs.from_dict(p) * self.coefs[k].lr)
                .to_dict()
                for k, p in self.best_coefs_data.items()
            },
            {
                k: self.tensors[k].scale_fn(p * self.tensors[k].lr)
                for k, p in self.best_tensors_data.items()
            },
        )
