"""Match models variables."""

import torch

from qgsw.fields.variables.uvh import UVH
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.specs import DEVICE
from qgsw.utils.dim_checks import with_dims


@with_dims(
    g_prime_nl=1,
    g_prime=1,
)
def match_psi(
    uvh_nl: UVH,
    g_prime_nl: torch.Tensor,
    g_prime: torch.Tensor,
) -> UVH:
    """Compute valid prognostic values.

    Ensure that the streamfunction of both models are the same.

    Args:
        uvh_nl (UVH): UVH from the reference model.
            ├── u: (n_ens, nl, nx+1, ny)-shaped
            ├── v: (n_ens, nl, nx, ny+1)-shaped
            └── h: (n_ens, nl, nx, ny)-shaped
        g_prime_nl (torch.Tensor): Reduced gravity for the reference model.
            └── (nl,)-shaped
        g_prime (torch.Tensor): Reduced gravity for the model
        to compute UVH for.
            └── (n,)-shaped

    Returns:
        UVH: Corresponding UVH.
            ├── u: (n_ens, n, nx+1, ny)-shaped
            ├── v: (n_ens, n, nx, ny+1)-shaped
            └── h: (n_ens, n, nx, ny)-shaped
    """
    u_nl, v_nl, h_nl = uvh_nl
    n = g_prime.shape[0]
    u = u_nl[:, :n, ...]
    v = v_nl[:, :n, ...]
    B = (  # noqa: N806
        g_prime_nl[None, :n, None, None]
        * reverse_cumsum(h_nl, dim=1)[:, :n, ...]
    )
    A = torch.ones((n, n), dtype=torch.float64, device=DEVICE.get()).triu()  # noqa: N806
    A *= g_prime.unsqueeze(-1)  # noqa: N806
    h = torch.einsum("lp,...pxy->...lxy", torch.linalg.inv(A), B)
    return UVH(u, v, h)
