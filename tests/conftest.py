"""Tests."""

import argparse

import pytest

from qgsw.specs import DEVICE

pytest_plugins = ["tests.fixtures"]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Pytest arg parser."""
    parser.addoption(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use CPU for tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure device."""
    if config.getoption("--cpu"):
        DEVICE.use_cpu()
