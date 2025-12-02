"""QG model wrapper base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgsw.models.qg.psiq.core import QGPSIQCore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import torch

    from qgsw.spatial.core.discretization import SpaceDiscretization2D

T = TypeVar("T", bound=QGPSIQCore)


class ModelWrapper(ABC, Generic[T]):
    """Model wrapper."""

    model: T
    color: str
    linestyle: str = "-"
    prefix: str | None
    label: str
    no_wind = False
    results_paths: Path
    cycle = -1
    save_params = False

    @property
    def plot_kwargs(self) -> dict[str, Any]:
        """Plotting kwargs."""
        return {
            "color": self.color,
            "linestyle": self.linestyle,
            "label": self.label,
        }

    def __init__(self, space_2d: SpaceDiscretization2D) -> None:
        """Instantiate the wrapper."""
        self.tracked = False
        self._init_model(space_2d)

    @abstractmethod
    def _init_model(self, space_2d: SpaceDiscretization2D) -> T: ...

    @abstractmethod
    def _set_params(self) -> None: ...

    @abstractmethod
    def compute_q(
        self, psi: torch.Tensor, beta_effect: torch.Tensor
    ) -> torch.Tensor:
        """Compute potential vorticity."""

    def set_wind_forcing(self, tx: torch.Tensor, ty: torch.Tensor) -> None:
        """Set wind forcing."""
        if self.no_wind:
            return
        self.model.set_wind_forcing(tx, ty)

    def new_cycle(self) -> None:
        """Starts new cycle."""
        self.cycle += 1

    @abstractmethod
    def setup(
        self,
        psis: list[torch.Tensor],
        times: list[torch.Tensor],
        beta_effect_w: torch.Tensor,
    ) -> None:
        """Setup model."""

    def step(self) -> None:
        """Perform a model step."""
        self.model.step()


M = TypeVar("M", bound=ModelWrapper[QGPSIQCore])


class ModelsManager(Generic[M]):
    """Manage multiple models at once."""

    def __init__(self, *mw: M) -> None:
        """Instantiate the manager."""
        self.model_wrappers = mw

        self.save_params = self.model_wrappers[0].save_params
        self.loop_over_models(lambda mw: setattr(mw, "tracked", True))

    @property
    def save_params(self) -> bool:
        """Whether to save model parameters."""
        return self.model_wrappers[0].save_params

    @save_params.setter
    def save_params(self, save_params: bool) -> None:
        self.loop_over_models(
            lambda mw: setattr(mw, "save_params", save_params)
        )

    def loop_over_models(self, func: Callable[[M], None]) -> None:
        """Iterate function over all model wrappers."""
        for mw in self.model_wrappers:
            func(mw)

    def step(self) -> None:
        """Perform a model step."""
        self.loop_over_models(lambda mw: mw.step())

    def new_cycle(self) -> None:
        """Start new cycle."""
        self.loop_over_models(lambda mw: mw.new_cycle())

    def reset_time(self) -> None:
        """Reset all model times."""
        self.loop_over_models(lambda mw: mw.model.reset_time())

    def set_wind_forcing(self, tx: torch.Tensor, ty: torch.Tensor) -> None:
        """Set wind forcing."""
        self.loop_over_models(lambda mw: mw.set_wind_forcing(tx, ty))

    def setup(
        self,
        psis: list[torch.Tensor],
        times: list[torch.Tensor],
        beta_effect_w: torch.Tensor,
    ) -> None:
        """Setup all models."""
        self.loop_over_models(lambda mw: mw.setup(psis, times, beta_effect_w))
