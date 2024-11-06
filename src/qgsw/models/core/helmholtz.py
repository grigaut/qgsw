# ruff: noqa
"""
Spectral 2D Helmholtz equation solver on rectangular and non-rectangular domain.
  - Colocated Dirichlet BC with DST-I  (type-I discrete sine transform)
  - Staggered Neumann   BC with DCT-II (type-II discrete consine transform)
  - Non-rectangular domains emmbedded in rectangular domains with a mask.
  - Capacitance matrix method for non-rectangular domains
Louis Thiry, 2023.
"""

import torch
import torch.nn.functional as F
from qgsw.specs import DEVICE


def compute_laplace_dctII(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    **kwargs,
) -> torch.Tensor:
    """DCT-II of standard 5-points laplacian on uniform grid"""
    x, y = torch.meshgrid(
        torch.arange(nx, **kwargs),
        torch.arange(ny, **kwargs),
        indexing="ij",
    )
    return (
        2 * (torch.cos(torch.pi / nx * x) - 1) / dx**2
        + 2 * (torch.cos(torch.pi / ny * y) - 1) / dy**2
    )


def dctII(x: torch.Tensor, exp_vec: torch.Tensor) -> torch.Tensor:
    """
    1D forward type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector exp_vec.
    """
    v = torch.cat([x[..., ::2], torch.flip(x, dims=(-1,))[..., ::2]], dim=-1)
    V = torch.fft.fft(v)
    return (V * exp_vec).real


def idctII(x: torch.Tensor, iexp_vec: torch.Tensor) -> torch.Tensor:
    """
    1D inverse type-II discrete cosine transform (DCT-II)
    using fft and precomputed auxillary vector iexp_vec.
    """
    N = x.shape[-1]
    x_rev = torch.flip(x, dims=(-1,))[..., :-1]
    v = (
        torch.cat(
            [x[..., 0:1], iexp_vec[..., 1:N] * (x[..., 1:N] - 1j * x_rev)],
            dim=-1,
        )
        / 2
    )
    V = torch.fft.ifft(v)
    y = torch.zeros_like(x)
    y[..., ::2] = V[..., : N // 2].real
    y[..., 1::2] = torch.flip(V, dims=(-1,))[..., : N // 2].real
    return y


def dctII2D(
    x: torch.Tensor, exp_vec_x: torch.Tensor, exp_vec_y: torch.Tensor
) -> torch.Tensor:
    """2D forward DCT-II."""
    return dctII(dctII(x, exp_vec_y).transpose(-1, -2), exp_vec_x).transpose(
        -1, -2
    )


def idctII2D(
    x: torch.Tensor, iexp_vec_x: torch.Tensor, iexp_vec_y: torch.Tensor
) -> torch.Tensor:
    """2D inverse DCT-II."""
    return idctII(
        idctII(x, iexp_vec_y).transpose(-1, -2), iexp_vec_x
    ).transpose(-1, -2)


def compute_dctII_exp_vecs(
    N: int,
    dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute auxillary exp_vec and iexp_vec used in
    fast DCT-II computations with FFTs."""
    N_range = torch.arange(N, dtype=dtype, device=device)
    exp_vec = 2 * torch.exp(-1j * torch.pi * N_range / (2 * N))
    iexp_vec = torch.exp(1j * torch.pi * N_range / (2 * N))
    return exp_vec, iexp_vec


def solve_helmholtz_dctII(
    rhs: torch.Tensor,
    helmholtz_dctII: torch.Tensor,
    exp_vec_x: torch.Tensor,
    exp_vec_y: torch.Tensor,
    iexp_vec_x: torch.Tensor,
    iexp_vec_y: torch.Tensor,
) -> torch.Tensor:
    """Solves Helmholtz equation with DCT-II fast diagonalisation."""
    rhs_dctII = dctII2D(rhs.type(helmholtz_dctII.dtype), exp_vec_x, exp_vec_y)
    return idctII2D(rhs_dctII / helmholtz_dctII, iexp_vec_x, iexp_vec_y).type(
        rhs.dtype
    )


def dstI1D(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """1D type-I discrete sine transform (DST-I), forward and inverse
    since DST-I is auto-inverse."""
    return torch.fft.irfft(-1j * F.pad(x, (1, 1)), dim=-1, norm=norm)[
        ..., 1 : x.shape[-1] + 1
    ]


def dstI2D(x: torch.Tensor, norm: str = "ortho"):
    """2D DST-I.

    2D Discrete Sine Transform.
    """
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(
        -1, -2
    )


def compute_laplace_dstI(
    nx: int, ny: int, dx: float, dy: float, **kwargs
) -> torch.Tensor:
    """Type-I discrete sine transform of the usual 5-points
    discrete laplacian operator on uniformly spaced grid.

    Laplacian in Fourier Space.
    """
    x, y = torch.meshgrid(
        torch.arange(1, nx, **kwargs),
        torch.arange(1, ny, **kwargs),
        indexing="ij",
    )
    return (
        2 * (torch.cos(torch.pi / nx * x) - 1) / dx**2
        + 2 * (torch.cos(torch.pi / ny * y) - 1) / dy**2
    )


def solve_helmholtz_dstI(
    rhs: torch.Tensor,
    helmholtz_dstI: torch.Tensor,
) -> torch.Tensor:
    """Solves 2D Helmholtz equation with DST-I fast diagonalization.

    Steps:
    - Perform forward DST-I of rhs.
    - Divide by helmholtz_dstI (in fourier space).
    - Perform inverse DST-I to com eback to original space.
    """
    return F.pad(
        dstI2D(dstI2D(rhs.type(helmholtz_dstI.dtype)) / helmholtz_dstI),
        (1, 1, 1, 1),
    ).type(rhs.dtype)


def compute_capacitance_matrices(
    helmholtz_dstI: torch.Tensor,
    bound_xids,
    bound_yids,
) -> torch.Tensor:
    nl = helmholtz_dstI.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    inv_cap_matrices = torch.zeros(
        (nl, M, M), dtype=torch.float64, device=DEVICE
    )
    rhs = torch.zeros(
        helmholtz_dstI.shape[-3:],
        dtype=torch.float64,
        device=helmholtz_dstI.device,
    )
    for m in range(M):
        rhs.fill_(0)
        rhs[..., bound_xids[m], bound_yids[m]] = 1
        sol = dstI2D(dstI2D(rhs) / helmholtz_dstI.type(torch.float64))
        inv_cap_matrices[:, m] = sol[..., bound_xids, bound_yids].to(
            device=DEVICE
        )

    # invert G matrices to get capacitance matrices
    cap_matrices = torch.zeros_like(inv_cap_matrices)
    for l in range(nl):
        cap_matrices[l] = torch.linalg.inv(inv_cap_matrices[l])

    return cap_matrices.to(helmholtz_dstI.device)


def solve_helmholtz_dstI_cmm(
    rhs: torch.Tensor,
    helmholtz_dstI: torch.Tensor,
    cap_matrices: torch.Tensor,
    bound_xids: torch.Tensor,
    bound_yids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    sol_1 = dstI2D(
        dstI2D(rhs.type(helmholtz_dstI.dtype)) / helmholtz_dstI
    ).type(rhs.dtype)
    alphas = torch.einsum(
        "...ij,...j->...i", cap_matrices, -sol_1[..., bound_xids, bound_yids]
    )

    rhs_2 = torch.zeros_like(rhs)
    rhs_2[..., bound_xids, bound_yids] = alphas

    return solve_helmholtz_dstI(rhs + rhs_2, helmholtz_dstI) * mask


class HelmholtzNeumannSolver:
    def __init__(self, nx, ny, dx, dy, lambd, dtype, device, mask=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.lambd = lambd
        self.device = device
        self.dtype = dtype

        # helmholtz dct-II
        self.helmholtz_dctII = (
            compute_laplace_dctII(nx, ny, dx, dy, dtype=dtype, device=device)
            - lambd
        )

        # auxillary vectors for DCT-II computations
        exp_vec_x, iexp_vec_x = compute_dctII_exp_vecs(nx, dtype, device)
        exp_vec_y, iexp_vec_y = compute_dctII_exp_vecs(ny, dtype, device)
        self.exp_vec_x = exp_vec_x.unsqueeze(0).unsqueeze(0)
        self.iexp_vec_x = iexp_vec_x.unsqueeze(0).unsqueeze(0)
        self.exp_vec_y = exp_vec_y.unsqueeze(0).unsqueeze(0)
        self.iexp_vec_y = iexp_vec_y.unsqueeze(0).unsqueeze(0)

        # mask
        if mask is not None:
            shape = mask.shape[0], mask.shape[1]
            assert shape == (
                nx,
                ny,
            ), f"Invalid mask {shape=} != nx, ny {nx, ny}"
            self.mask = mask.unsqueeze(0).type(dtype).to(device)
        else:
            self.mask = torch.ones(
                1, nx, ny, dtype=self.dtype, device=self.device
            )
        self.not_mask = 1 - self.mask

        # mask on u- and v-grid
        self.mask_u = (
            F.avg_pool2d(
                self.mask,
                (2, 1),
                stride=(1, 1),
                padding=(1, 0),
                divisor_override=1,
            )
            > 1.5
        ).type(self.dtype)
        self.mask_v = (
            F.avg_pool2d(
                self.mask,
                (1, 2),
                stride=(1, 1),
                padding=(0, 1),
                divisor_override=1,
            )
            > 1.5
        ).type(self.dtype)

        # irregular boundary indices
        mask_neighbor_x = self.mask * F.pad(
            F.avg_pool2d(self.mask, (3, 1), stride=(1, 1), divisor_override=1)
            < 2.5,
            (0, 0, 1, 1),
        )
        mask_neighbor_y = self.mask * F.pad(
            F.avg_pool2d(self.mask, (1, 3), stride=(1, 1), divisor_override=1)
            < 2.5,
            (1, 1, 0, 0),
        )
        self.mask_irrbound = torch.logical_or(mask_neighbor_x, mask_neighbor_y)
        self.irrbound_xids, self.irrbound_yids = torch.where(
            self.mask_irrbound[0]
        )

        # compute capacitance matrix
        self.compute_capacitance_matrix()

    def helmholtz_reg_domain(self, f):
        f_ = F.pad(f, (1, 1, 1, 1), mode="replicate")
        dxx_f = (
            f_[..., 2:, 1:-1] + f_[..., :-2, 1:-1] - 2 * f_[..., 1:-1, 1:-1]
        ) / self.dx**2
        dyy_f = (
            f_[..., 1:-1, 2:] + f_[..., 1:-1, :-2] - 2 * f_[..., 1:-1, 1:-1]
        ) / self.dy**2
        return dxx_f + dyy_f - self.lambd * f

    def helmholtz(self, f):
        if len(self.irrbound_xids) == 0:
            return self.helmholtz_reg_domain(f)

        f_ = F.pad(f, (1, 1, 1, 1), mode="replicate")
        dx_f = torch.diff(f_[..., 1:-1], dim=-2) / self.dx
        dy_f = torch.diff(f_[..., 1:-1, :], dim=-1) / self.dy
        dxx_f = torch.diff(dx_f * self.mask_u, dim=-2) / self.dx
        dyy_f = torch.diff(dy_f * self.mask_v, dim=-1) / self.dy

        return (dxx_f + dyy_f - self.lambd * f) * self.mask

    def compute_capacitance_matrix(self):
        M = len(self.irrbound_xids)
        if M == 0:
            self.cap_matrix = None
            return

        # compute inverse capacitance matrice
        inv_cap_matrix = torch.zeros(
            (M, M), dtype=torch.float64, device=self.device
        )
        for m in range(M):
            v = torch.zeros(M, device=self.device, dtype=torch.float64)
            v[m] = 1
            inv_cap_matrix[:, m] = v - self.V_T(self.G(self.U(v)))

        # invert on cpu
        cap_matrix = torch.linalg.inv(inv_cap_matrix.to(device=DEVICE))

        # convert to dtype and transfer to device
        self.cap_matrix = cap_matrix.type(self.dtype).to(self.device)

    def U(self, v):
        Uv = torch.zeros_like(self.mask)
        Uv[..., self.irrbound_xids, self.irrbound_yids] = v
        return Uv

    def V_T(self, field):
        return (self.helmholtz_reg_domain(field) - self.helmholtz(field))[
            ..., self.irrbound_xids, self.irrbound_yids
        ]

    def G(self, field):
        return solve_helmholtz_dctII(
            field,
            self.helmholtz_dctII,
            self.exp_vec_x,
            self.exp_vec_y,
            self.iexp_vec_x,
            self.iexp_vec_y,
        )

    def solve(self, rhs):
        GF = self.G(rhs)
        if len(self.irrbound_xids) == 0:
            return GF
        V_TGF = self.V_T(GF)
        rho = torch.einsum("ij,...j->...i", self.cap_matrix, V_TGF)
        return GF + self.G(self.U(rho))
