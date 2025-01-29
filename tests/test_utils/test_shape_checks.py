"""Test shape checks."""

from typing import Any, ParamSpec

import pytest
import torch

from qgsw.specs import DEVICE
from qgsw.utils.shape_checks import ShapeValidationError, with_shapes

Param = ParamSpec("Param")

default_y = torch.rand(
    (5,),
    dtype=torch.float64,
    device=DEVICE.get(),
)


@with_shapes(
    x=(5,),
    y=(5,),
)
def func(
    x: torch.Tensor,  # noqa: ARG001
    y: torch.Tensor = default_y,  # noqa: ARG001
    *args: Param.args,  # noqa: ARG001
    **kwargs: Param.kwargs,  # noqa: ARG001
) -> None:
    """Random function.

    Args:
        x (torch.Tensor): Tensor.
        y (torch.Tensor, optional): Tensor. Defaults to default_y.
        *args (Param.args): Arguments.
        **kwargs (Param.kwargs): Keyword arguments.
    """
    return


testdata = [
    pytest.param((torch.ones((5, 1)),), {}),
    pytest.param((torch.ones((5, 1)),), {"z": 1}),
    pytest.param((torch.ones((5, 1)), torch.ones((5,))), {}),
    pytest.param((torch.ones((5)), torch.ones((5, 1))), {}),
    pytest.param((), {"x": torch.ones((5, 1))}),
    pytest.param((), {"x": torch.ones((5, 1)), "y": torch.ones((5,))}),
    pytest.param((), {"x": torch.ones((5,)), "y": torch.ones((5, 1))}),
]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    testdata,
)
def test_with_shapes_errors(args: Any, kwargs: Any) -> None:  # noqa: ANN401
    """Ensure with_shapes raises errors."""
    with pytest.raises(ShapeValidationError):
        func(*args, **kwargs)


testdata = [
    pytest.param((torch.ones((5,)),), {}),
    pytest.param((torch.ones((5,)),), {"z": 1}),
    pytest.param((torch.ones((5,)), torch.ones((5,))), {}),
    pytest.param((), {"x": torch.ones((5,))}),
    pytest.param((), {"x": torch.ones((5,)), "y": torch.ones((5,))}),
]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    testdata,
)
def test_with_shapes_valid(args: Any, kwargs: Any) -> None:  # noqa: ANN401
    """Ensure with_shapes does not raise errors."""
    try:
        func(*args, **kwargs)
    except ShapeValidationError as err:
        raise AssertionError from err
