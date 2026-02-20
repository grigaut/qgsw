"""Space Configuration tests."""

import pytest
import torch

from qgsw.configs.space import SpaceConfig
from qgsw.spatial.core.discretization import SpaceDiscretization2D


@pytest.fixture
def space_config() -> SpaceConfig:
    """Space configuration."""
    return SpaceConfig(
        nx=110,
        ny=62,
        unit="m",
        x_min=-50,
        x_max=500,
        y_min=0,
        y_max=124,
    )


def test_dx_dy_ds(
    space_config: SpaceConfig,
) -> None:
    """Test that dx,dy and ds are consistent."""
    space_2d = SpaceDiscretization2D.from_config(space_config)
    assert torch.isclose(space_config.dx, space_2d.dx).all()
    assert torch.isclose(space_config.dy, space_2d.dy).all()
    assert torch.isclose(space_config.ds, space_2d.ds).all()
