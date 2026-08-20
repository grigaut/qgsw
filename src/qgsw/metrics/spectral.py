"""Spectral metrics."""

from __future__ import annotations

from math import ceil, sqrt

import torch

from qgsw.specs import defaults


class CrossSpectralDensity:
    """Cross spectral density (CSD)."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        *,
        truncate: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate cross spectral density.

        Args:
            nx (int): Domainsize along x.
            ny (int): Domain size along y.
            dx (float): Spacing along x.
            dy (float): Spacing along y.
            truncate (bool, optional): Whether to truncate spectrum for
                wavenumbers larger than the Nyquist wavenumber.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self.specs = defaults.get(dtype=dtype, device=device)
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy

        # Create horizontal wavenumbers
        kx = torch.fft.fftfreq(self.nx, self.dx / (2 * torch.pi))
        ky = torch.fft.fftfreq(self.ny, self.dy / (2 * torch.pi))

        kx, ky = torch.meshgrid(
            kx.to(**self.specs), ky.to(**self.specs), indexing="ij"
        )
        self.kh = torch.sqrt(kx**2 + ky**2).flatten()

        # Create isotropic wavenumbers
        self.dkx, self.dky = (
            2 * torch.pi / (self.nx * self.dx),
            2 * torch.pi / (self.ny * self.dy),
        )
        self.dkr = max(self.dkx, self.dky)  # spacing
        if truncate:
            self.nkr = ceil(0.5 * max(self.nx / 2, self.ny / 2))
        else:
            self.nkr = ceil(sqrt(2) * max(self.nx / 2, self.ny / 2))

        self.kr = self.dkr * torch.arange(1, self.nkr + 1, **self.specs)

        # Scaling factor of energy density due to DFT and integration
        self.fft_scaling = self.dx * self.dy
        self.density_scaling = (self.dkx / (2 * torch.pi)) * (
            self.dky / (2 * torch.pi)
        )

        # Window for Hann windowing
        self.window, self.window_scaling = self._create_window(
            self.nx, self.ny
        )

    def _create_window(
        self, nx: int, ny: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create 2D Hann window."""
        window_x = torch.windows.hann(nx, sym=False, **self.specs)
        window_y = torch.windows.hann(ny, sym=False, **self.specs)

        # Produit extérieur pour obtenir la matrice 2D (forme: nx, ny)
        window = torch.outer(window_x, window_y)
        window_scaling = torch.mean(window**2)
        return window, window_scaling

    def _check_sizes(self, f: torch.Tensor) -> None:
        assert f.shape[-2] == self.nx, "Incorrect size in x-axis"
        assert f.shape[-1] == self.ny, "Incorrect size in y-axis"

    def isotropize(self, fhat: torch.Tensor) -> torch.Tensor:
        """Integration between lower and upper bounds of annular rings."""
        fhat_ = fhat.flatten(start_dim=-2, end_dim=-1)
        fpsd = torch.zeros([*fhat_.shape[:-1], self.nkr], **self.specs)
        for p in range(self.nkr):
            idk = (self.kh >= self.kr[p] - self.dkr / 2) & (
                self.kh < self.kr[p] + self.dkr / 2
            )
            fpsd[..., p] = torch.sum(fhat_[..., idk], dim=-1)
        return fpsd

    def spec(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        *,
        scale_fft: bool = True,
    ) -> torch.Tensor:
        """Compute CSD of 2D fields f1 and f2.

        Args:
            f1 (torch.Tensor): First Tensor.
            f2 (torch.Tensor): Second Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.

        Returns:
            torch.Tensor: Cross spectral density of f1 and f2.
        """
        self._check_sizes(f1)
        self._check_sizes(f2)

        fhat1: torch.Tensor = torch.fft.fft2(f1, dim=(-2, -1))
        fhat2: torch.Tensor = torch.fft.fft2(f2, dim=(-2, -1))

        if scale_fft:
            return (
                fhat1
                * torch.conj(fhat2)
                * self.density_scaling
                * self.fft_scaling**2
            )
        return fhat1 * torch.conj(fhat2) * self.density_scaling

    def spec_w(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        *,
        scale_fft: bool = True,
        window_correction: bool = False,
    ) -> torch.Tensor:
        """Compute (windowed) CSD of 2D fields f1 and f2.

        Args:
            f1 (torch.Tensor): First Tensor.
            f2 (torch.Tensor): Second Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Cross spectral density of f1 and f2.
        """
        self._check_sizes(f1)
        self._check_sizes(f2)

        fhat1: torch.Tensor = torch.fft.fft2(f1 * self.window)
        fhat2: torch.Tensor = torch.fft.fft2(f2 * self.window)

        fhat: torch.Tensor = fhat1 * torch.conj(fhat2)

        scaling = self.window_scaling if window_correction else 1

        if scale_fft:
            return fhat * self.density_scaling * self.fft_scaling**2 / scaling

        return fhat * self.density_scaling / scaling

    def iso_spec(
        self, f1: torch.Tensor, f2: torch.Tensor, *, scale_fft: bool = True
    ) -> torch.Tensor:
        """Compute isotropic CSD of 2D fields f1 and f2.

        Args:
            f1 (torch.Tensor): First Tensor.
            f2 (torch.Tensor): Second Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.

        Returns:
            torch.Tensor: Cross spectral density of f1 and f2.
        """
        return self.isotropize(self.spec(f1=f1, f2=f2, scale_fft=scale_fft))

    def iso_spec_w(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        *,
        scale_fft: bool = True,
        window_correction: bool = False,
    ) -> torch.Tensor:
        """Compute (windowed) CSD of 2D fields f1 and f2.

        Args:
            f1 (torch.Tensor): First Tensor.
            f2 (torch.Tensor): Second Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Cross spectral density of f1 and f2.
        """
        return self.isotropize(
            self.spec_w(
                f1=f1,
                f2=f2,
                scale_fft=scale_fft,
                window_correction=window_correction,
            )
        )


class PowerSpectralDensity:
    """Compute Power Spectral Density (PSD) of scalar 2D field."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate PowerSpectralDensity.

        Args:
            nx (int): Domainsize along x.
            ny (int): Domain size along y.
            dx (float): Spacing along x.
            dy (float): Spacing along y.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self.csd = CrossSpectralDensity(
            nx, ny, dx, dy, dtype=dtype, device=device
        )

    @property
    def kr(self) -> torch.Tensor:
        """Frequencies."""
        return self.csd.kr

    def isotropize(self, fhat: torch.Tensor) -> torch.Tensor:
        """Integration between lower and upper bounds of annular rings."""
        return self.csd.isotropize(fhat=fhat)

    def spec(self, f: torch.Tensor, *, scale_fft: bool = True) -> torch.Tensor:
        """Compute (windowed) power spectral density of 2D field f.

        Args:
            f (torch.Tensor): Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.

        Returns:
            torch.Tensor: Power spectral density of f.
        """
        return self.csd.spec(f, f, scale_fft=scale_fft)

    def spec_w(
        self,
        f: torch.Tensor,
        *,
        scale_fft: bool = True,
        window_correction: bool = False,
    ) -> torch.Tensor:
        """Compute (windowed) power spectral density of 2D field f.

        Args:
            f (torch.Tensor): Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Power spectral density of f.
        """
        return self.csd.spec_w(
            f,
            f,
            scale_fft=scale_fft,
            window_correction=window_correction,
        )

    def iso_spec(
        self,
        f: torch.Tensor,
        *,
        scale_fft: bool = True,
    ) -> torch.Tensor:
        """Compute isotropic PSD of 2D field f.

        Args:
            f (torch.Tensor): Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Power spectral density of f.
        """
        return self.isotropize(self.spec(f=f, scale_fft=scale_fft))

    def iso_spec_w(
        self,
        f: torch.Tensor,
        *,
        scale_fft: bool = True,
        window_correction: bool = False,
    ) -> torch.Tensor:
        """Compute isotropic (windowed) power spectral density of 2D field f.

        Args:
            f (torch.Tensor): Tensor.
            scale_fft (bool, optional): Whether to apply scaling to FFT or not.
                Defaults to True.
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Power spectral density of f.
        """
        return self.isotropize(
            self.spec_w(
                f=f,
                scale_fft=scale_fft,
                window_correction=window_correction,
            )
        )


class MagnitudeSquaredCoherence:
    """Magnitude Squared Coherence (MSC)."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: int,
        dy: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate MagnitudeSquaredCoherence.

        Args:
            nx (int): Domainsize along x.
            ny (int): Domain size along y.
            dx (float): Spacing along x.
            dy (float): Spacing along y.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self.csd = CrossSpectralDensity(
            nx, ny, dx, dy, dtype=dtype, device=device
        )

    @property
    def kr(self) -> torch.Tensor:
        """Frequencies."""
        return self.csd.kr

    def iso_spec(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSC of 2D field f.

        Args:
            f1 (torch.Tensor): First tensor.
            f2 (torch.Tensor): Second tensor.

        Returns:
            torch.Tensor: Magnitude squared coherence of f1 and f2.
        """
        csd_f1f2 = self.csd.spec(f1, f2, scale_fft=False)
        psd_f1 = self.csd.iso_spec(f1, f1, scale_fft=False)
        psd_f2 = self.csd.iso_spec(f2, f2, scale_fft=False)

        return self.csd.isotropize(torch.abs(csd_f1f2) ** 2) / (
            psd_f1 * psd_f2
        )

    def iso_spec_w(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        *,
        window_correction: bool = False,
    ) -> torch.Tensor:
        """Compute (windowed) MSC of 2D field f.

        Args:
            f1 (torch.Tensor): First tensor.
            f2 (torch.Tensor): Second tensor.
            dim (tuple[int, int], optional): Dimensions for FFT.
                Defaults to (-2, -1).
            window_correction (bool, optional): Whether to apply window
                correction or not. Defaults to False.

        Returns:
            torch.Tensor: Magnitude squared coherence of f1 and f2.
        """
        csd_f1f2 = self.csd.iso_spec_w(
            f1,
            f2,
            scale_fft=False,
            window_correction=window_correction,
        )
        psd_f1 = self.csd.iso_spec_w(
            f1,
            f1,
            scale_fft=False,
            window_correction=window_correction,
        )
        psd_f2 = self.csd.iso_spec_w(
            f2,
            f2,
            scale_fft=False,
            window_correction=window_correction,
        )

        return torch.abs(csd_f1f2) ** 2 / (psd_f1 * psd_f2)
