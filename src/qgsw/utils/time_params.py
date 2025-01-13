"""Compute time-related parameters for runs."""

import torch

from qgsw.fields.variables.uvh import PrognosticTuple
from qgsw.spatial.core.discretization import SpaceDiscretization3D
from qgsw.specs import DEVICE

CFL_ADV = 0.5
CFL_GRAVITY = 0.5


def compute_dt(
    prognostic: PrognosticTuple,
    space: SpaceDiscretization3D,
    g_prime: torch.Tensor,
    h: torch.Tensor,
) -> float:
    """Compute optimal dt.

    Args:
        prognostic (PrognosticTuple): Prognostic Values.
        space (SpaceDiscretization3D): 3D Space Discretization.
        g_prime (torch.Tensor): Reduced Gravity.
        h (torch.Tensor): Layers Thickness.

    Returns:
        float: Timestep in s.
    """
    u = prognostic.u
    v = prognostic.v
    u_max, v_max, c = (
        torch.abs(u).max().item() / space.dx,
        torch.abs(v).max().item() / space.dy,
        torch.sqrt(g_prime[0] * h.sum()),
    )

    dt = min(
        CFL_ADV * space.dx / u_max,
        CFL_ADV * space.dy / v_max,
        CFL_GRAVITY * space.dx / c,
    )
    return dt.cpu().item()


def compute_tau(omega: torch.Tensor, space: SpaceDiscretization3D) -> float:
    """Compute optimal tau value.

    Args:
        omega (torch.Tensor): Vorticity.
        space (SpaceDiscretization3D): 3D Space.

    Returns:
        float: Tau, in seconds.
    """
    w = omega.squeeze() / space.ds
    return 1.0 / torch.sqrt(w.pow(2).mean()).to(device=DEVICE.get()).item()
