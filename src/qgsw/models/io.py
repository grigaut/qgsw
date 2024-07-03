"""Model Data retriever."""

from pathlib import Path

import numpy as np
import torch

from qgsw import verbose
from qgsw.models.exceptions import InvalidSavingFileError
from qgsw.models.variables import UVH
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization3D


class ModelResultsRetriever:
    """Model Results Retriever."""

    space: SpaceDiscretization3D
    uvh: UVH
    device: str
    omega: torch.Tensor
    p: torch.Tensor
    beta_plane: BetaPlane
    eta: torch.Tensor

    def get_physical_uvh(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        u_phys = (self.uvh.u / self.space.dx).to(device=self.device)
        v_phys = (self.uvh.v / self.space.dy).to(device=self.device)
        h_phys = (self.uvh.h / self.space.area).to(device=self.device)

        return (u_phys, v_phys, h_phys)

    def get_physical_uvh_as_ndarray(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: u, v and h
        """
        return (e.cpu().numpy() for e in self.get_physical_uvh())

    def get_physical_omega(
        self,
    ) -> torch.Tensor:
        """Get physical vorticity.

        Returns:
            torch.Tensor: Vorticity
        """
        vorticity = self.omega / self.space.area / self.beta_plane.f0
        return vorticity.to(device=self.device)

    def get_physical_omega_as_ndarray(
        self,
    ) -> np.ndarray:
        """Get physical vorticity.

        Returns:
            np.ndarray: Vorticity
        """
        return self.get_physical_omega().cpu().numpy()

    def get_print_info(self) -> str:
        """Returns a string with summary of current variables.

        Returns:
            str: Summary of variables.
        """
        hl_mean = (self.uvh.h / self.space.area).mean((-1, -2))
        eta = self.eta
        u = self.uvh.u / self.space.dx
        v = self.uvh.v / self.space.dy
        h = self.uvh.h / self.space.area
        with np.printoptions(precision=2):
            eta_surface = eta[:, 0].min().to(device=self.device).item()
            return (
                f"u: {u.mean().to(device=self.device).item():+.5E}, "
                f"{u.abs().max().to(device=self.device).item():.5E}, "
                f"v: {v.mean().to(device=self.device).item():+.5E}, "
                f"{v.abs().max().to(device=self.device).item():.5E}, "
                f"hl_mean: {hl_mean.squeeze().cpu().numpy()}, "
                f"h min: {h.min().to(device=self.device).item():.5E}, "
                f"max: {h.max().to(device=self.device).item():.5E}, "
                f"eta_sur min: {eta_surface:+.5f}, "
                f"max: {eta[:,0].max().to(device=self.device).item():.5f}"
            )

    def _raise_if_invalid_savefile(self, output_file: Path) -> None:
        """Raise and error if the saving file is invalid.

        Args:
            output_file (Path): Output file.

        Raises:
            InvalidSavingFileError: if the saving file extension is not .npz.
        """
        if output_file.suffix != ".npz":
            msg = "Variables are expected to be saved in an .npz file."
            raise InvalidSavingFileError(msg)

    def save_uvh(self, output_file: Path) -> None:
        """Save U, V and H values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        u, v, h = self.get_physical_uvh_as_ndarray()

        np.savez(
            output_file,
            u=u.astype("float32"),
            v=v.astype("float32"),
            h=h.astype("float32"),
        )

        verbose.display(msg=f"saved u,v,h to {output_file}", trigger_level=1)

    def save_omega(self, output_file: Path) -> None:
        """Save vorticity values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        omega = self.get_physical_omega_as_ndarray()

        np.savez(output_file, omega=omega.astype("float32"))

        verbose.display(msg=f"saved ω to {output_file}", trigger_level=1)

    def save_uvhwp(self, output_file: Path) -> None:
        """Save uvh, vorticity and pressure values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        omega = self.get_physical_omega_as_ndarray()
        u, v, h = self.get_physical_uvh_as_ndarray()

        np.savez(
            output_file,
            u=u.astype("float32"),
            v=v.astype("float32"),
            h=h.astype("float32"),
            omega=omega.astype("float32"),
            p=self.p.cpu().numpy().astype("float32"),
        )

        verbose.display(
            msg=f"saved u,v,h,ω,p to {output_file}",
            trigger_level=1,
        )
