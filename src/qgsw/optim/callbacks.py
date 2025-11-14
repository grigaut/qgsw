"""Callback for optimization."""

import torch

from qgsw.logging import getLogger

logger = getLogger(__name__)


class LRChangeCallback:
    """Callback for learning rates.

    For better logs, requires the parameters to have a 'name'
    field in the optimizer.
    """

    name_key = "name"

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        """Instantiate the callback.

        Args:
            optimizer (torch.optim.Optimizer): optimizer to track.
        """
        self._optimizer = optimizer

        self._lrs = self._create_lr_dict()

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

    def step(self) -> None:
        """Log a message if the learning rate changes."""
        lrs = self._create_lr_dict()
        for k, v in lrs.items():
            if v == (v_ := self._lrs[k]):
                continue
            action = "decreased" if v < v_ else "increased"

            msg = f"[{k}] Learning rate {action} from {v_} to {v}."
            logger.info(msg)
        self._lrs = lrs
