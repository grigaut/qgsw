"""Test shape checks."""

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from typing import Any

import pytest
import torch

from qgsw.specs import DEVICE
from qgsw.utils.shape_checks import (
    ShapeValidationError,
    with_shapes,
    with_variable_shapes,
)
from qgsw.utils.size import Size

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


sizex = Size(1)
sizey = Size(1)


@with_variable_shapes(
    x=(sizex + 1,),
    y=(sizey,),
)
def func2(x: torch.Tensor, y: torch.Tensor) -> None:  # noqa: ARG001
    """Random function.

    Args:
        x (torch.Tensor): Tensor.
        y (torch.Tensor): Tensor.
    """
    return


testdata = [
    pytest.param(5, 5, torch.ones((6, 1)), torch.ones((5,))),
    pytest.param(5, 5, torch.ones(5), torch.ones((5, 1))),
    pytest.param(5, 5, torch.ones(4), torch.ones((5, 1))),
    pytest.param(5, 5, torch.ones(5), torch.ones(5)),
    pytest.param(5, 5, torch.ones(5), torch.ones(5)),
]


@pytest.mark.parametrize(
    ("size_x", "size_y", "x", "y"),
    testdata,
)
def test_variable_shape_errors(
    size_x: int,
    size_y: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Ensure with_variable_shapes raises errors."""
    sizex.update(size_x)
    sizey.update(size_y)
    with pytest.raises(ShapeValidationError):
        func2(x, y)


testdata = [
    pytest.param(5, 5, torch.ones((6,)), torch.ones((5,))),
    pytest.param(10, 4, torch.ones((11,)), torch.ones((4,))),
]


@pytest.mark.parametrize(
    ("size_x", "size_y", "x", "y"),
    testdata,
)
def test_variable_shape_valid(
    size_x: int,
    size_y: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Ensure with_variable_shapes raises errors."""
    sizex.update(size_x)
    sizey.update(size_y)
    try:
        func2(x, y)
    except ShapeValidationError as err:
        raise AssertionError from err
