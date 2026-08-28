"""Boundaries extraction methods."""

import torch

from qgsw.solver.boundary_conditions.base import Boundaries


def extract_psi_bc(psi: torch.Tensor, bc: int) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2)


def extract_q_bc(q: torch.Tensor, bc: int) -> Boundaries:
    """Extract q."""
    return Boundaries.extract(q, bc - 2, -(bc - 1), bc - 2, -(bc - 1), 3)


def extract_sst_bc(sst: torch.Tensor, bc: int) -> Boundaries:
    """Extract SST."""
    return Boundaries.extract(sst, bc - 1, -bc, bc - 1, -bc, 3)
