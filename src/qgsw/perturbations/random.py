"""Random perturbation."""

import torch
from torch.nn import functional as F  # noqa: N812

from qgsw import verbose
from qgsw.mesh.mesh import Mesh3D
from qgsw.perturbations.base import _Perturbation
from qgsw.specs import DEVICE


class RandomSurfacePerturbation(_Perturbation):
    """Random perturbation."""

    _type = "random-uniform"
    _concerned_area_ratio: int = 2
    _kernel_size_ratio: int = 5
    _sigma: float = 3

    def __init__(self, magnitude: float = 0.001) -> None:
        """Instantiate RandomSurfacePerturbation.

        Args:
            magnitude (float, optional): Perturbation Magnitude.
            Defaults to 0.001.
        """
        super().__init__(magnitude)

    def compute_scale(self, mesh: Mesh3D) -> float:
        """Compute the scale of the perturbation.

        The scale refers to the typical size of the perturbation, not its
        magnitude. For example the scale of a vortex pertubation would be its
        radius.

        Args:
            mesh (Mesh3D): 3D Mesh.

        Returns:
            float: Perturbation scale.
        """
        # Real size of the gaussian kernel window
        return mesh.lx / self._kernel_size_ratio

    def _generate_random_field(
        self,
        nx: int,
        ny: int,
    ) -> torch.Tensor:
        """Generate a random field in the center of the area.

        Args:
            nx (int): Number of cells in the x direction
            ny (int): Number of cells in the y direction

        Returns:
            torch.Tensor: Random field.
        """
        # Select a sub area of the entire grid
        nx_rand_area = nx // self._concerned_area_ratio
        ny_rand_area = ny // self._concerned_area_ratio

        random_field = torch.rand(
            (nx_rand_area, ny_rand_area),
            dtype=torch.float64,
            device=DEVICE,
        )  # shape (nx_rand_area, ny_rand_area)
        centered_random_field = 2 * random_field - 1
        # pad random field to match grid size.
        pad_x_left = (nx - ny_rand_area) // 2
        pad_x_right = nx - pad_x_left - nx_rand_area
        pad_y_left = (ny - ny_rand_area) // 2
        pad_y_right = ny - pad_y_left - ny_rand_area
        pad = (
            pad_x_left,
            pad_x_right,
            pad_y_left,
            pad_y_right,
        )
        return F.pad(centered_random_field, pad, value=0)  # shape: (nx, ny)

    def _generate_gaussian_kernel(
        self,
        size: int,
    ) -> torch.Tensor:
        """Generate the gaussian kernel.

        Args:
            size (int): 'Radius' of the kernel.

        Returns:
            torch.Tensor: (2*size, 2*size) gaussian kernel.
        """
        mean = (size - 1) / 2.0
        variance = self._sigma**2.0

        x_cord = torch.arange(size, dtype=torch.float64, device=DEVICE)
        x_grid = x_cord.repeat(size).view(size, size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * torch.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance),
        )
        # Make sure sum of values in gaussian kernel equals 1.
        return gaussian_kernel / torch.sum(gaussian_kernel)

    def compute_stream_function(self, mesh: Mesh3D) -> torch.Tensor:
        """Compute the stream function induced by the perturbation.

        Args:
            mesh (Mesh3D): 3D Mesh.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        kernel_size = int(mesh.nx / self._kernel_size_ratio)
        size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        random_field = self._generate_random_field(
            nx=mesh.nx,  # + (size - 1),
            ny=mesh.ny,  # + (size - 1),
        )
        verbose.display(
            f"Random perturbation: Gaussian kernel size: {size}",
            trigger_level=2,
        )
        kernel = self._generate_gaussian_kernel(size=size)
        psi_2d = F.conv2d(
            random_field.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding="same",
        )  # shape: (1,1,nx,ny)
        nx, ny = psi_2d.shape[-2:]
        psi = torch.ones(
            (1, mesh.nl, nx, ny),
            device=DEVICE,
            dtype=torch.float64,
        )

        psi[0, 0, ...] = psi_2d
        return psi  # shape: (1, nl, nx, ny)
