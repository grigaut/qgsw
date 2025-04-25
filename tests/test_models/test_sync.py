"""Model synchronization tests."""

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import torch

from qgsw.configs.space import SpaceConfig
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.core import QG
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sw.core import SW
from qgsw.models.synchronization import ModelSync
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import defaults
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.models.base import ModelUVH


@pytest.fixture
def space2d() -> SpaceDiscretization2D:
    """2D Space."""
    config = SpaceConfig(
        nx=15,
        ny=15,
        unit=Unit.M,
        x_min=0,
        x_max=150,
        y_min=0,
        y_max=150.0,
    )
    return SpaceDiscretization2D.from_config(config)


@pytest.fixture
def qg_3l(space2d: SpaceDiscretization2D) -> QG:
    """Three layers QG model."""
    return QG(
        space_2d=space2d,
        H=torch.tensor([400, 1100, 2600], **defaults.get()),
        g_prime=torch.tensor([9.81, 0.025, 0.0125], **defaults.get()),
        beta_plane=BetaPlane(f0=9.375e-5, beta=0),
        optimize=True,
    )


@pytest.fixture
def sw_3l(space2d: SpaceDiscretization2D) -> SW:
    """Three layers SW model."""
    return SW(
        space_2d=space2d,
        H=torch.tensor([400, 1100, 2600], **defaults.get()),
        g_prime=torch.tensor([9.81, 0.025, 0.0125], **defaults.get()),
        beta_plane=BetaPlane(f0=9.375e-5, beta=0),
        optimize=True,
    )


@pytest.fixture
def qg_1l(space2d: SpaceDiscretization2D) -> QG:
    """One layer QG model."""
    return QG(
        space_2d=space2d,
        H=torch.tensor([400], **defaults.get()),
        g_prime=torch.tensor([9.81], **defaults.get()),
        beta_plane=BetaPlane(f0=9.375e-5, beta=0),
        optimize=True,
    )


testdata = [
    pytest.param(
        "qg_3l",
        "qg_1l",
        id="QG-3L -> QG-1L",
    ),
    pytest.param(
        "sw_3l",
        "qg_1l",
        id="SW-3L -> QG-1L",
    ),
]


@pytest.mark.parametrize(("model_ref_str", "model_str"), testdata)
def test_model_sync(
    model_ref_str: str,
    model_str: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test model synchronization.

    Args:
        model_ref_str (str): Model ref fixture name.
        model_str (str): Model fixture name.
        request (pytest.FixtureRequest): Fixture request.
    """
    model_ref: ModelUVH = request.getfixturevalue(model_ref_str)
    model: ModelUVH = request.getfixturevalue(model_str)
    model_ = deepcopy(model)
    nl = model.space.nl
    model_sync = ModelSync(model_ref, model)

    p0 = torch.zeros((1, 3, 16, 16), **defaults.get())
    p0[:, :, 8:-8, 8:-8] = 1

    model_ref.set_p(p0)

    model_sync()

    P = QGProjector(  # noqa: N806
        compute_A(
            model_ref.H[:, 0, 0],
            model_ref.g_prime[:, 0, 0],
            **defaults.get(),
        ),
        model_ref.H,
        model_ref.space,
        model_ref.beta_plane.f0,
        model_ref.masks,
    )
    p = P.compute_p(model_ref.prognostic.uvh)[0]
    model_.set_p(p[:, :nl])

    torch.testing.assert_close(model_.u, model.u)
    torch.testing.assert_close(model_.v, model.v)
    torch.testing.assert_close(model_.h, model.h)
