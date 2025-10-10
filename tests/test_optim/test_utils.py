"""Test optim utils."""

import torch

from qgsw.models.qg.psiq.optim.utils import EarlyStop, RegisterParams


def test_early_stop() -> None:
    """Test early stopping."""
    early_stop = EarlyStop(eps=1e-1)
    has_stopped = False
    for n in range(1, 500):
        loss = torch.tensor(1 / n)
        if early_stop.step(loss):
            has_stopped = True
            break
    assert has_stopped


def test_register_params() -> None:
    """Test parameter registration."""
    register_params = RegisterParams()
    for n in range(1, 500):
        param = torch.tensor(n)
        loss = (param / 250 - 1) ** 2
        register_params.step(loss, n=param)
    assert "n" in register_params.params
    assert register_params.params["n"] == 250  # noqa: PLR2004
    assert register_params.best_loss == 0
