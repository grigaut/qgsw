"""Create a Baroclinic Vortex."""
# ruff: noqa: PGH004
# ruff: noqa

import torch
from torch.nn import functional as F  # noqa: N812

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def grad_perp(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Orthogonal gradient."""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


def compute_laplace_dstI(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    **kwargs,
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


def dstI1D(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """1D type-I discrete sine transform (DST-I), forward and inverse
    since DST-I is auto-inverse.
    """
    return torch.fft.irfft(-1j * F.pad(x, (1, 1)), dim=-1, norm=norm)[
        ...,
        1 : x.shape[-1] + 1,
    ]


def dstI2D(x: torch.Tensor, norm: str = "ortho"):
    """2D DST-I.

    2D Discrete Sine Transform.
    """
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(
        -1,
        -2,
    )


class BaroclinicVortex:
    """Baroclinic Vortex."""

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int,
        ny: int,
    ) -> None:
        """Instantiate BaroclinicVortex.

        Args:
            x_min (float): X minimum bound.
            x_max (float): X maximum bound.
            y_min (float): Y minimum bound.
            y_max (float): Y maximum bound.
            nx (int): Number of point in the X direction.
            ny (int): Number of point in the Y direction.
        """
        self.lx = x_max - x_min
        self.ly = y_max - y_min
        self.nx = nx
        self.ny = ny
        self.dx = self.lx / nx
        self.dy = self.ly / ny
        self.x, self.y = torch.meshgrid(
            torch.linspace(
                x_min,
                x_max,
                nx,
                dtype=torch.float64,
                device=DEVICE,
            ),
            torch.linspace(
                y_min,
                y_max,
                ny,
                dtype=torch.float64,
                device=DEVICE,
            ),
            indexing="ij",
        )

    @property
    def r0(self) -> float:
        """Inner circle radius."""
        return 0.1 * self.lx

    @property
    def r1(self) -> float:
        """Outer circle first border radius."""
        return self.r0

    @property
    def r2(self) -> float:
        """Outer circle second border radius."""
        return 0.14 * self.lx

    def compute_vorticity(
        self,
    ) -> torch.Tensor:
        """Compute the vorticity."""
        # Compute cylindrical components
        theta = torch.angle(self.x + 1j * self.y)
        r = torch.sqrt(self.x**2 + self.y**2)
        r = r * (1 + 0.001 * torch.cos(theta * 3))
        # Mask vortex's core
        mask_core = torch.sigmoid((self.r0 - r) / 100)
        # Mask vortex's ring
        inner_ring = torch.sigmoid((r - self.r1) / 100)
        outer_ring = torch.sigmoid((self.r2 - r) / 100)
        mask_ring = inner_ring * outer_ring
        # compute vorticity
        vortex = mask_ring / mask_ring.mean() - mask_core / mask_core.mean()
        return vortex / vortex.abs().max()

    def compute_stream_function_2d(self) -> torch.Tensor:
        """Compute the value of the streamfunction ψ.

        Returns:
            torch.Tensor: Streamfunction values over the domain,
            (1, 1, nx, ny)-shaped..
        """
        vor = self.compute_vorticity()
        # Compute Laplacian operator in Fourier Space
        laplacian = compute_laplace_dstI(
            self.nx - 1,
            self.ny - 1,
            self.dx,
            self.dy,
            device=DEVICE,
            dtype=torch.float64,
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(dstI2D(psi_hat), (1, 1, 1, 1))
        return psi.unsqueeze(0).unsqueeze(0)

    def compute_stream_function(self, nl: int) -> torch.Tensor:
        """Value of the stream function ψ.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        psi = torch.ones(
            (1, nl, self.nx, self.ny),
            device=DEVICE,
            dtype=torch.float64,
        )
        psi_2d = self.compute_stream_function_2d()
        psi[0, 0, ...] = psi_2d
        return psi

    def adjust_stream_function(
        self,
        psi: torch.Tensor,
        f0: float,
        Ro: float,
    ) -> torch.Tensor:
        """Adjust stream function values to match Rossby's number.

        Args:
            psi (torch.Tensor): Stream function.
            f0 (float): Coriolis Parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Stream function, (1, nl, nx, ny)-shaped.
        """
        u, v = grad_perp(psi, self.dx, self.dy)
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        return psi * (Ro * f0 * self.r0 / u_norm_max)

    def convert_to_pressure(
        self,
        psi: torch.Tensor,
        f0: float,
    ) -> torch.Tensor:
        """Convert stream function to pressure.

        Args:
            psi (torch.Tensor): Stream function.
            f0 (float): Coriolis Parameter.

        Returns:
            torch.Tensor: Pressure, (1, nl, nx, ny)-shaped.
        """
        return psi * f0

    def compute_initial_pressure(
        self,
        f0: float,
        Ro: float,
    ) -> torch.Tensor:
        """Compute the initial pressure values.

        Args:
            grid_3d (Grid3D): 3D Grid.
            f0 (float): Coriolis parameter.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Pressure values.
        """
        psi = self.compute_stream_function(nl=2)
        psi_adjusted = self.adjust_stream_function(psi, f0, Ro)
        return self.convert_to_pressure(psi=psi_adjusted, f0=f0)


if __name__ == "__main__":
    import plotly.express as px
    import plotly.graph_objects as go

    vortex = BaroclinicVortex(
        -500_000,
        500_000,
        -500_000,
        500_000,
        192,
        192,
    )

    f0 = 9.375e-5
    Ro = 0.1

    X = torch.linspace(-10, 10, 192).repeat((192, 1))

    X = torch.full((192, 192), 0)

    sf = vortex.compute_stream_function(2)
    sf = sf / torch.max(torch.abs(sf))

    omega = torch.diff(
        torch.diff(F.pad(sf, (1, 1, 0, 0)), dim=-1),
        dim=-1,
    ) + torch.diff(
        torch.diff(F.pad(sf, (0, 0, 1, 1)), dim=-2),
        dim=-2,
    )

    cmax = torch.max(torch.abs(omega)).cpu().item()

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            z=omega[0, 0],
            colorscale=px.colors.diverging.RdBu_r,
            cmin=-cmax,
            cmax=cmax,
        ),
    )
    offset = 1
    fig.add_trace(
        go.Surface(
            z=omega[0, 1] - offset,
            colorscale=px.colors.diverging.RdBu_r,
            cmin=-cmax - offset,
            cmax=cmax - offset,
            showscale=False,
        ),
    )

    fig.show()
