"""Test initial condition."""

from typing import TYPE_CHECKING

import pytest

from qgsw.models.synchronization.initial_conditions import InitialCondition
from qgsw.specs import defaults

if TYPE_CHECKING:
    from qgsw.models.base import ModelUVH

testdata = [
    pytest.param("QG_model", id="QG"),
    pytest.param("SW_model", id="SW"),
]


@pytest.mark.parametrize(("model_fixture"), testdata)
def test_steady(model_fixture: str, request: pytest.FixtureRequest) -> None:
    """Test steady initial condition."""
    model: ModelUVH = request.getfixturevalue(model_fixture)
    ic = InitialCondition(model)
    ic.set_steady(**defaults.get())

    assert (model.u == 0).all()
    assert (model.v == 0).all()
    assert (model.h == 0).all()
