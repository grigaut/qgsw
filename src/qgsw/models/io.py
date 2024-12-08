"""Model Data retriever."""

from pathlib import Path

import numpy as np
import torch

from qgsw import verbose
from qgsw.models.exceptions import InvalidSavingFileError
from qgsw.models.variables import UVH
from qgsw.specs._utils import Device


class ModelResultsRetriever:
    """Model Results Retriever."""

    uvh_phys: UVH
    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor
    device: Device
    omega: torch.Tensor
    omega_phys: torch.Tensor
    p: torch.Tensor
    eta: torch.Tensor

    def get_physical_uvh(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get physical variables u_phys, v_phys, h_phys from state variables.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u, v and h
        """
        u_phys = self.uvh_phys.u.to(device=self.device.get())
        v_phys = self.uvh_phys.v.to(device=self.device.get())
        h_phys = self.uvh_phys.h.to(device=self.device.get())

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
        return self.omega_phys.to(
            device=self.device.get(),
        )  # divide by self.batea_plane.f0 ?

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
        hl_mean = (self.uvh_phys.h).mean((-1, -2))
        eta = self.eta
        u = self.uvh_phys.u
        v = self.uvh_phys.v
        h = self.uvh_phys.h
        device = self.device.get()
        with np.printoptions(precision=2):
            eta_surface = eta[:, 0].min().to(device=device).item()
            return (
                f"u: {u.mean().to(device=device).item():+.5E}, "
                f"{u.abs().max().to(device=device).item():.5E}, "
                f"v: {v.mean().to(device=device).item():+.5E}, "
                f"{v.abs().max().to(device=device).item():.5E}, "
                f"hl_mean: {hl_mean.squeeze().cpu().numpy()}, "
                f"h min: {h.min().to(device=device).item():.5E}, "
                f"max: {h.max().to(device=device).item():.5E}, "
                f"eta_sur min: {eta_surface:+.5f}, "
                f"max: {eta[:,0].max().to(device=device).item():.5f}"
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

        verbose.display(
            msg=f"saved physical u,v,h to {output_file}",
            trigger_level=1,
        )

    def save_omega(self, output_file: Path) -> None:
        """Save vorticity values.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)

        omega = self.get_physical_omega_as_ndarray()

        np.savez(output_file, omega=omega.astype("float32"))

        verbose.display(
            msg=f"saved physical vorticity to {output_file}",
            trigger_level=1,
        )

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
