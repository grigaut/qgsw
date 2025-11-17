"""Fast Fourier Transforms tools in Pytorch."""

import torch
import torch.nn.functional as F  # noqa: N812


def dstI1Dffreq(  # noqa: N802
    n: int,
    d: float = 1,
    *,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Wraps torch.fft.fftfreq.

    Args:
        n (int): Input tensor shape.
        d (float, optional): The sampling length scale.
        The spacing between individual samples of the FFT input.
        The default assumes unit spacing, dividing that result by the actual
        spacing gives the result in physical frequency units. Defaults to 1.0.
        dtype (torch.dtype, optional): dtype. Defaults to None.
        device (torch.device, optional): device. Defaults to None.

    Returns:
        torch.Tensor: torch.fft.ffreq(2*n,d)
    """
    return torch.fft.fftfreq(
        n=2 * n,
        d=d,
        dtype=dtype,
        device=device,
    )[:n]


def dstI1D(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:  # noqa: N802
    """1D type-I discrete sine transform (DST-I).

    Forward and inverse since DST-I is auto-inverse.

    irfft(x) <=> ifft([x[0],...,x[-2],x[-1],x[-2],...,x[1]])
    irfft(-1j*F.pad(x,(1,1))) <=>
    ifft([0,-1j*x[0],...,-1j*x[-1],0,-1j*x[-1],...,-1j*x[0]])
    ifft([0,-1j*x[0],...,-1j*x[-1],0,-1j*x[-1],...,-1j*x[0]]) =
    """
    return torch.fft.irfft(-1j * F.pad(x, (1, 1)), dim=-1, norm=norm)[
        ...,
        1 : x.shape[-1] + 1,
    ]


def dstI2D(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:  # noqa: N802
    """2D DST-I.

    2D Discrete Sine Transform.
    """
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(
        -1,
        -2,
    )


def dctII(x: torch.Tensor, exp_vec: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """1D forward type-II discrete cosine transform (DCT-II).

    Using fft and precomputed auxillary vector exp_vec.
    """
    v = torch.cat([x[..., ::2], torch.flip(x, dims=(-1,))[..., ::2]], dim=-1)
    V = torch.fft.fft(v)
    return (V * exp_vec).real


def idctII(x: torch.Tensor, iexp_vec: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """1D inverse type-II discrete cosine transform (DCT-II).

    Using fft and precomputed auxillary vector iexp_vec.
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


def dctII2D(  # noqa: N802
    x: torch.Tensor,
    exp_vec_x: torch.Tensor,
    exp_vec_y: torch.Tensor,
) -> torch.Tensor:
    """2D forward DCT-II."""
    return dctII(dctII(x, exp_vec_y).transpose(-1, -2), exp_vec_x).transpose(
        -1,
        -2,
    )


def idctII2D(  # noqa: N802
    x: torch.Tensor,
    iexp_vec_x: torch.Tensor,
    iexp_vec_y: torch.Tensor,
) -> torch.Tensor:
    """2D inverse DCT-II."""
    return idctII(
        idctII(x, iexp_vec_y).transpose(-1, -2),
        iexp_vec_x,
    ).transpose(-1, -2)


def compute_dctII_exp_vecs(  # noqa: N802
    N: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute auxillary exp_vec and iexp_vec.

    Used in fast DCT-II computations with FFTs.
    """
    N_range = torch.arange(N, dtype=dtype, device=device)
    exp_vec = 2 * torch.exp(-1j * torch.pi * N_range / (2 * N))
    iexp_vec = torch.exp(1j * torch.pi * N_range / (2 * N))
    return exp_vec, iexp_vec
