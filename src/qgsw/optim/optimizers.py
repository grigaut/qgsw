"""Optimizers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch

from qgsw.logging.core import getLogger
from qgsw.logging.utils import step
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop
from qgsw.utils.named_object import Name, NamedObject

ClosureInput = TypeVar("ClosureInput")
ClosureOutput = TypeVar("ClosureOutput")

logger = getLogger(__name__)


class EarlyStopError(Exception):
    """Exception for early stop."""


class OptimizerName(Name):
    """Optimizer names."""

    LBFGS = "lbfgs"
    ADAM = "adam"


class OptimizerWrapper(
    NamedObject[OptimizerName],
    ABC,
    Generic[ClosureInput, ClosureOutput],
):
    """Optimizer base class."""

    optimizer: torch.optim.Optimizer
    _pre_closure: Callable[[], ClosureInput]
    _post_closure: Callable[[ClosureOutput], torch.Tensor]
    epoch = 0
    max_epoch: int | None = None

    @property
    def current_epoch(self) -> list[tuple[str, int]]:
        """Current epoch."""
        return [("epoch", self.epoch)]

    @property
    def epoch_repr(self) -> str:
        """Current epoch string representation."""
        return ", ".join([f"{n} {v + 1}" for n, v in self.current_epoch])

    @abstractmethod
    def __init__(
        self,
        params: list[torch.Tensor],
        losses: dict[str, list[float]] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Instantiate the optimizer.

        Args:
            params (list[torch.Tensor]): params.
            losses (dict[str, list[float]] | None, optional): Losses.
                Defaults to None.
            **kwargs (Any): Any kwargs.
        """
        msg = f"Instantiating: {self.get_type().value} optimizer."
        logger.info(msg)
        self.losses: dict[str, list[float]] = (
            {
                "obs": [],
                "dyn": [],
                "bkg": [],
                "best": [],
            }
            if losses is None
            else losses
        )

    def _step_no_closure(self) -> torch.Tensor:
        msg = "Set closure using set_closure first"
        raise NotImplementedError(msg)

    def _step(self) -> torch.Tensor:
        """Make step."""
        self.optimizer.step(closure=self.closure)
        self.epoch += 1

    step = _step_no_closure

    def set_closure(
        self, closure: Callable[[ClosureInput], ClosureOutput]
    ) -> None:
        """Set the closure function.

        Args:
            closure (Callable[[ClosureInput], ClosureOutput]): Closure
                function.
        """
        self._set_closure(closure, self._pre_closure, self._post_closure)

    def _set_closure(
        self,
        closure: Callable[[ClosureInput], ClosureOutput],
        pre_closure_callback: Callable[[], ClosureInput] | None = None,
        post_closure_callback: Callable[[ClosureOutput], torch.Tensor]
        | None = None,
    ) -> None:

        def optim_closure() -> torch.Tensor:
            closure_input = pre_closure_callback()
            closure_output = closure(closure_input)
            return post_closure_callback(closure_output)

        self.closure = optim_closure
        self.step = self._step


class LBFGS(OptimizerWrapper[None, dict[str, list[torch.Tensor]]]):
    """LBFGS optimizer wrapper."""

    _type = OptimizerName.LBFGS

    @property
    def max_eval(self) -> int:
        """Max number of evaluations."""
        return self.optimizer.param_groups[0]["max_eval"]

    @property
    def current_epoch(self) -> list[tuple[str, int]]:
        """Current epoch."""
        return [*super().current_epoch, ("evaluation", self.eval)]

    def __init__(
        self,
        params: list[torch.Tensor],
        max_iter: int = 20,
        history_size: int = 20,
        losses: dict[str, list[torch.Tensor]] | None = None,
    ) -> None:
        """Instantiate LBFGS optimizer wrapper.

        Args:
            params (list[torch.Tensor]): Parameters to optim.
            max_iter (int, optional): Max number of iterations. Defaults to 20.
            history_size (int, optional): History size. Defaults to 20.
            losses (dict[str, list[torch.Tensor]] | None, optional): Losses.
                Defaults to None.
        """
        super().__init__(
            params, losses, max_iter=max_iter, history_size=history_size
        )

        self.optimizer = torch.optim.LBFGS(
            params,
            lr=1,
            line_search_fn="strong_wolfe",
            history_size=history_size,
            max_iter=max_iter,
        )
        self.early_stop = EarlyStop()
        self.eval = 0

        def pre_closure() -> None:
            self.optimizer.zero_grad()

        def post_closure(losses: dict[str, torch.Tensor]) -> torch.Tensor:

            tot = losses["tot"].detach()
            obs = losses["obs"].detach().item()
            dyn = losses["dyn"].detach().item()
            bkg = losses["bkg"].detach().item()
            best = losses["best"].detach().item()

            msg = (
                f"[Epoch {step(self.epoch + 1, self.max_epoch)}]"
                f"[Eval {step(self.eval + 1, self.max_eval)}]"
                f"[Best loss: {best:>#10.5g}]"
                f" Loss: {tot.item():>#10.5g}"
                f" | Obs: {obs:>#10.5g}"
                f" | Dyn Reg: {dyn:>#10.5g}"
                f" | Bkg Reg: {bkg:>#10.5g}"
            )
            logger.info(msg)
            self.losses["obs"].append(obs)
            self.losses["dyn"].append(dyn)
            self.losses["bkg"].append(bkg)
            self.losses["best"].append(best)
            if self.early_stop.step(tot):
                msg = "Convergence reached"
                raise EarlyStopError(msg)
            self.eval += 1
            return tot

        self._pre_closure = pre_closure
        self._post_closure = post_closure

    def _step(self) -> torch.Tensor:
        self.eval = 0
        return super()._step()


class Adam(OptimizerWrapper[None, dict[str, list[torch.Tensor]]]):
    """Adam optimizer wrapper."""

    _type = OptimizerName.ADAM

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1,
        losses: dict[str, list[torch.Tensor]] | None = None,
    ) -> None:
        """Instantiate the optimizer.

        Args:
            params (list[torch.Tensor]): _description_
            lr (float, optional): _description_. Defaults to 1.
            losses (dict[str, list[torch.Tensor]] | None, optional): losses.
                Defaults to None.
        """
        super().__init__(params, losses, lr=lr)
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5
        )
        self.lr_callback = LRChangeCallback(self.optimizer)
        self.early_stop = EarlyStop()

        def pre_closure() -> None:
            self.optimizer.zero_grad()

        def post_closure(losses: dict[str, torch.Tensor]) -> torch.Tensor:

            tot = losses["tot"].detach()
            obs = losses["obs"].detach().item()
            dyn = losses["dyn"].detach().item()
            bkg = losses["bkg"].detach().item()
            best = losses["best"].detach().item()

            msg = (
                f"[Epoch {step(self.epoch + 1, self.max_epoch)}]"
                f"[Best loss: {best:>#10.5g}]"
                f" Loss: {tot.item():>#10.5g}"
                f" | Obs: {obs:>#10.5g}"
                f" | Dyn Reg: {dyn:>#10.5g}"
                f" | Bkg Reg: {bkg:>#10.5g}"
            )
            logger.info(msg)
            self.losses["obs"].append(obs)
            self.losses["dyn"].append(dyn)
            self.losses["bkg"].append(bkg)
            self.losses["best"].append(best)
            if self.early_stop.step(tot):
                msg = "Convergence reached"
                raise EarlyStopError(msg)
            self.scheduler.step(tot)
            self.lr_callback.step()
            return tot

        self._pre_closure = pre_closure
        self._post_closure = post_closure


def build_optimizer(
    optimizer_type: OptimizerName, params: list[torch.Tensor]
) -> OptimizerWrapper:
    """Build optimizer.

    Args:
        optimizer_type (OptimizerName): Optimizer type.
        params (list[torch.Tensor]): PArams.

    Raises:
        ValueError: If the optimizer type is invalid.

    Returns:
        Optimizer: Optimizer.
    """
    match optimizer_type:
        case OptimizerName.LBFGS:
            return LBFGS(params, max_iter=20, history_size=20)
        case OptimizerName.ADAM:
            return Adam(params, lr=1)
        case _:
            msg = f"Unrecognized optimizer type: {optimizer_type}."
            raise ValueError(msg)
