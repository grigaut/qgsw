"""Callback for optimization."""

import torch

from qgsw import logging

logger = logging.getLogger(__name__)


class LRChangeCallback:
    """Callback for learning rates.

    For better logs, requires the parameters to have a 'name'
    field in the optimizer.
    """

    name_key = "name"

    def __init__(
        self, optimizer: torch.optim.Optimizer, level: int = logging.INFO
    ) -> None:
        """Instantiate the callback.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to track.
            level (int): Logging level.
        """
        self._optimizer = optimizer
        self.level = level

        self._lrs = self._create_lr_dict()
        self.initial_log()

    def _create_lr_dict(self) -> dict[str, float]:
        """Create learning rate dictionnary.

        Returns:
            dict[str, float]: Parameter name -> Learning rate.
        """
        key = self.name_key
        return {
            param.get(key, f"Parameter {i}"): param["lr"]
            for i, param in enumerate(self._optimizer.param_groups)
        }

    def initial_log(self) -> None:
        """Initial log."""
        for k, v in self._lrs.items():
            msg = f"[{k}] Learning rate set to {v}."
            logger.log(self.level, msg)

    def step(self) -> None:
        """Log a message if the learning rate changes."""
        lrs = self._create_lr_dict()
        for k, v in lrs.items():
            if v == (v_ := self._lrs[k]):
                continue
            action = "decreased" if v < v_ else "increased"

            msg = f"[{k}] Learning rate {action} from {v_} to {v}."
            logger.log(self.level, msg)
        self._lrs = lrs
