"""Match models variables."""

import torch

from qgsw.fields.variables.prognostic_tuples import UVH
from qgsw.specs import defaults
from qgsw.utils.dim_checks import with_dims


@with_dims(
    g_prime_nl=1,
    g_prime=1,
)
def match_psi(
    uvh_nl: UVH,
    g_prime_nl: torch.Tensor,
    g_prime: torch.Tensor,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> UVH:
    """Compute prognostic vars to match stream function between models.

    The reference streamfunction is supposed to have nl levels and the output
    is expected to have n layers (nl >= n).

    Ensure that the streamfunction of both models are the same.
    ψ_nl = M_nl @ h_nl / f_0 with M_nl: (nl, nl)-shaped
    ψ    = M    @ h    / f_0 with M   : (n , n )-shaped
    --> u = u_nl[:n]                (Omitting non-layer dimensions)
    --> v = v_nl[:n]                (Omitting non-layer dimensions)
    --> h = M⁻¹ @ ( M_nl @ h)[:n]   (Omitting non-layer dimensions)

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
        dtype (torch.dtype, optional): Dtype. Defaults to None.
        device (torch.device, optional): Device. Defaults to None.

    Returns:
        UVH: Corresponding UVH.
            ├── u: (n_ens, n, nx+1, ny)-shaped
            ├── v: (n_ens, n, nx, ny+1)-shaped
            └── h: (n_ens, n, nx, ny)-shaped
    """
    nl = g_prime_nl.shape[0]
    n = g_prime.shape[0]
    if nl < n:
        msg = f"n should be lower than nl (n: {n} !<= nl: {nl})"
        raise ValueError(msg)
    dtype = defaults.get_dtype(dtype)
    device = defaults.get_device(device)
    u_nl, v_nl, h_nl = uvh_nl
    u = u_nl[:, :n, ...]
    v = v_nl[:, :n, ...]

    ones_nl = torch.ones((nl, nl), dtype=dtype, device=device)
    B_nl = ones_nl.tril() * g_prime_nl.unsqueeze(0)  # noqa: N806
    M_nl = B_nl @ ones_nl.triu()  # noqa: N806

    ones = torch.ones((n, n), dtype=dtype, device=device)
    B = ones.tril() * g_prime.unsqueeze(0)  # noqa: N806
    M = B @ ones.triu()  # noqa: N806

    # Compute pressure: p = (M_nl @ h_nl)[:n]
    p = torch.einsum("lp,...pxy->...lxy", M_nl, h_nl)[:, :n, ...]

    # Solve h = M⁻¹ @ ( M_nl @ h)[:n]
    h = torch.einsum(
        "lp,...pxy->...lxy",
        torch.linalg.inv(M),
        p,
    )
    return UVH(u, v, h)


def match_pv(uvh_nl: UVH, n: int) -> UVH:
    """Compute prognostic vars to match potential vorticity between models.

    The reference potential vorticity is supposed to have nl levels and the
    output is expected to have n layers (nl >= n).

    Ensure that the potential vorticity of both models are the same.
    q_nl = ∂_x v_nl - ∂_y u_nl - f_0 h_nl / H
    q    = ∂_x v    - ∂_y u    - f_0 h    / H
    --> u = u_nl[:n]                (Omitting non-layer dimensions)
    --> v = v_nl[:n]                (Omitting non-layer dimensions)
    --> h = h_nl[:n]                (Omitting non-layer dimensions)

    Args:
        uvh_nl (UVH): UVH from the reference model.
            ├── u: (n_ens, nl, nx+1, ny)-shaped
            ├── v: (n_ens, nl, nx, ny+1)-shaped
            └── h: (n_ens, nl, nx, ny)-shaped
        n (int): Output desired shape.

    Returns:
        UVH: Corresponding UVH.
            ├── u: (n_ens, n, nx+1, ny)-shaped
            ├── v: (n_ens, n, nx, ny+1)-shaped
            └── h: (n_ens, n, nx, ny)-shaped
    """
    nl = uvh_nl.u.shape[1]
    if nl < n:
        msg = f"n should be lower than nl (n: {n} !<= nl: {nl})"
        raise ValueError(msg)
    return UVH(
        uvh_nl.u[:, :n],
        uvh_nl.v[:, :n],
        uvh_nl.h[:, :n],
    )
