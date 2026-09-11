"""Boundaries extraction methods."""

import torch

from qgsw.solver.boundary_conditions.base import Boundaries


def extract_psi_bc(
    psi: torch.Tensor,
    bc: int,
    *,
    clone: bool = False,
) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2, clone=clone)


def extract_wide_psi_bc(
    psi_f: torch.Tensor,
    bc: int,
    *,
    clone: bool = False,
) -> Boundaries:
    """Extract wide psi boundaries."""
    return Boundaries.extract(
        psi_f,
        bc + 1,
        -bc - 2,
        bc + 1,
        -bc - 2,
        6,
        clone=clone,
    )


def extract_q_bc(
    q: torch.Tensor,
    bc: int,
    *,
    clone: bool = False,
) -> Boundaries:
    """Extract q."""
    return Boundaries.extract(
        q, bc - 2, -(bc - 1), bc - 2, -(bc - 1), 3, clone=clone
    )


def extract_sst_bc(
    sst: torch.Tensor,
    bc: int,
    *,
    clone: bool = False,
) -> Boundaries:
    """Extract SST."""
    return Boundaries.extract(sst, bc - 1, -bc, bc - 1, -bc, 3, clone=clone)
