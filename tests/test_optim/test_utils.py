"""Test optim utils."""

import torch

from qgsw.optim.utils import EarlyStop


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
