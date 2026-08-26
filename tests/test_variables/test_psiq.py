"""Test for PSIQ-derived tuples."""

from __future__ import annotations

import pytest
import torch

from qgsw.fields.variables.tuples import (
    PSIQ,
    PSIQSST,
    PSIQSSTT,
    PSIQT,
    BasePSIQ,
    BasePSIQSST,
    PSIQSSTTAlpha,
    PSIQTAlpha,
    rand_like,
)
from qgsw.specs import defaults


def gen_psi_q_sst_t_alpha() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Generate random psi, q, sst, t, alpha tensors."""
    n_ens = 3
    nl = 2
    nx = 20
    ny = 30
    psi = torch.rand(
        (n_ens, nl, nx + 1, ny),
        **defaults.get(),
    )
    q = torch.rand(
        (n_ens, nl, nx, ny + 1),
        **defaults.get(),
    )
    sst = torch.rand(
        (n_ens, nl, nx, ny),
        **defaults.get(),
    )
    t = torch.rand(1, **defaults.get()).squeeze()
    alpha = torch.rand(1, **defaults.get()).squeeze()
    return psi, q, sst, t, alpha


@pytest.fixture
def psiq() -> PSIQ:
    """PSIQ."""
    psi, q, _, _, _ = gen_psi_q_sst_t_alpha()
    return PSIQ(psi=psi, q=q)


@pytest.fixture
def psiqsst() -> PSIQSST:
    """PSIQ."""
    psi, q, sst, _, _ = gen_psi_q_sst_t_alpha()
    return PSIQSST(psi=psi, q=q, sst=sst)


@pytest.fixture
def psiqt() -> PSIQT:
    """PSIQ."""
    psi, q, _, t, _ = gen_psi_q_sst_t_alpha()
    return PSIQT(psi=psi, q=q, t=t)


@pytest.fixture
def psiqsstt() -> PSIQSSTT:
    """PSIQ."""
    psi, q, sst, t, _ = gen_psi_q_sst_t_alpha()
    return PSIQSSTT(psi=psi, q=q, sst=sst, t=t)


@pytest.fixture
def psiqtalpha() -> PSIQTAlpha:
    """PSIQ."""
    psi, q, _, t, alpha = gen_psi_q_sst_t_alpha()
    return PSIQTAlpha(psi=psi, q=q, t=t, alpha=alpha)


@pytest.fixture
def psiqssttalpha() -> PSIQSSTTAlpha:
    """PSIQ."""
    psi, q, sst, t, alpha = gen_psi_q_sst_t_alpha()
    return PSIQSSTTAlpha(psi=psi, q=q, sst=sst, t=t, alpha=alpha)


testdata = [
    pytest.param("psiq", id="psiq"),
    pytest.param("psiqt", id="psiqt"),
    pytest.param("psiqtalpha", id="psiqtalpha"),
]
testdatasst = [
    pytest.param("psiqsst", id="psiqsst"),
    pytest.param("psiqsstt", id="psiqsstt"),
    pytest.param("psiqssttalpha", id="psiqssttalpha"),
]


@pytest.mark.parametrize(("prognostic"), testdata + testdatasst)
def test_rand_like(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test rand_like function."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    rand_prognostic = rand_like(base_prognostic)
    assert isinstance(rand_prognostic, base_prognostic.__class__)
    for f in base_prognostic._fields:
        assert (
            getattr(rand_prognostic, f).shape
            == getattr(base_prognostic, f).shape
        )


@pytest.mark.parametrize(("prognostic"), testdata + testdatasst)
def test_psiq_attr(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test .psiq attribute."""
    base_psiq: BasePSIQ | BasePSIQSST = request.getfixturevalue(prognostic)
    assert base_psiq.psiq == PSIQ(base_psiq.psi, base_psiq.q)


@pytest.mark.parametrize(("prognostic"), testdatasst)
def test_psiqsst_attr(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test .psiq attribute."""
    base_psiqsst: BasePSIQSST = request.getfixturevalue(prognostic)
    assert base_psiqsst.psiqsst == PSIQSST(
        base_psiqsst.psi, base_psiqsst.q, base_psiqsst.sst
    )


@pytest.mark.parametrize(("prognostic"), testdata)
def test_add(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test addition of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    rand_prognostic = rand_like(base_prognostic)
    added_prognostic = base_prognostic + rand_prognostic
    for f in base_prognostic.psiq._fields:
        torch.testing.assert_close(
            getattr(added_prognostic, f),
            (getattr(base_prognostic, f) + getattr(rand_prognostic, f)),
        )


@pytest.mark.parametrize(("prognostic"), testdatasst)
def test_add_sst(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test addition of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    rand_prognostic = rand_like(base_prognostic)
    added_prognostic = base_prognostic + rand_prognostic
    for f in base_prognostic.psiqsst._fields:
        torch.testing.assert_close(
            getattr(added_prognostic, f),
            (getattr(base_prognostic, f) + getattr(rand_prognostic, f)),
        )


@pytest.mark.parametrize(("prognostic"), testdata)
def test_sub(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test subtraction of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    rand_prognostic = rand_like(base_prognostic)
    subtracted_prognostic = base_prognostic - rand_prognostic
    for f in base_prognostic.psiq._fields:
        torch.testing.assert_close(
            getattr(subtracted_prognostic, f),
            (getattr(base_prognostic, f) - getattr(rand_prognostic, f)),
        )


@pytest.mark.parametrize(("prognostic"), testdatasst)
def test_sub_sst(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test subtraction of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    rand_prognostic = rand_like(base_prognostic)
    subtracted_prognostic = base_prognostic - rand_prognostic
    for f in base_prognostic.psiqsst._fields:
        torch.testing.assert_close(
            getattr(subtracted_prognostic, f),
            (getattr(base_prognostic, f) - getattr(rand_prognostic, f)),
        )


@pytest.mark.parametrize(("prognostic"), testdata)
def test_mul(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test multiplication of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    multiplied_prognostic = base_prognostic * 5
    for f in base_prognostic.psiq._fields:
        torch.testing.assert_close(
            getattr(multiplied_prognostic, f),
            (getattr(base_prognostic, f) * 5),
        )


@pytest.mark.parametrize(("prognostic"), testdatasst)
def test_mul_sst(prognostic: str, request: pytest.FixtureRequest) -> None:
    """Test multiplication of prognostic tuples."""
    base_prognostic: BasePSIQ | BasePSIQSST = request.getfixturevalue(
        prognostic
    )
    multiplied_prognostic = base_prognostic * 5
    for f in base_prognostic.psiqsst._fields:
        torch.testing.assert_close(
            getattr(multiplied_prognostic, f),
            (getattr(base_prognostic, f) * 5),
        )
