"""Utils."""

import torch


class EarlyStop:
    """Early stop."""

    def __init__(self, *, eps: float = 1e-8, stop_after: int = 10) -> None:
        """INstantiate EarlyStop.

        Args:
            eps (float, optional): Precision. Defaults to 1e-8.
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
            self.previous_loss = loss
            return False
        loss_ = self.previous_loss
        if ((loss - loss_).abs() / loss_.abs()) < self.eps:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.stop_after


class RegisterParams:
    """Keep track of lower-loss parameters."""

    def __init__(self) -> None:
        """Instantiate RegisterParams."""
        self.best_loss = torch.tensor(float("inf"))
        self.params = None

    def step(self, loss: torch.Tensor, **kwargs: torch.Tensor) -> None:
        """Check against new loss.

        Args:
            loss (torch.Tensor): Loss value.
            **kwargs (torch.Tensor): Parameters.
        """
        if loss > self.best_loss:
            return
        self.params = {e: kwargs[e].detach() for e in kwargs}
        self.best_loss = torch.clone(loss.detach())

    def __repr__(self) -> str:
        """Implement __repr__."""
        return f"Best loss: {self.best_loss:.5f}, with params: \n{self.params}"
