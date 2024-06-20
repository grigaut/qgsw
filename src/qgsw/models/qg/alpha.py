"""Compute Colinearity Coefficient."""

from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
from scipy import interpolate
from typing_extensions import Self

ABOVE_ZERO_THRESHOLD = 1e-5


def mean_threshold(array: np.ndarray, std_nb: int = 1) -> float:
    """Compute Threshold Based on mean value.

    Threshold is 'mean + std_nb * std', computed on absolute values.

    Args:
        array (np.ndarray): Array to base threshold on.
        std_nb (int, optional): Number of standard deviation to add.
        Defaults to 1.

    Returns:
        float: mean + std_nb * std
    """
    array_abs = np.abs(array)
    mean = np.mean(array_abs[array_abs > ABOVE_ZERO_THRESHOLD])
    std = np.std(array_abs[array_abs > ABOVE_ZERO_THRESHOLD])
    return mean + std_nb * std


def quantile_threshold(
    array: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """Compute Threshold based on quantile.

    Args:
        array (np.ndarray): Array to base threshold on.
        quantile (float, optional): Quantile value. Defaults to 0.95.

    Returns:
        float: Threshold: quantile value over basolute values.
    """
    array_abs = np.abs(array)
    abs_top = array_abs[array_abs > ABOVE_ZERO_THRESHOLD]
    return np.quantile(abs_top, quantile)


def compute_coef_from_two_layers(
    top: np.ndarray,
    bottom: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute the coefficient using layers values.

    Args:
        top (np.ndarray): Top layer.
        bottom (np.ndarray): Bottom layer.
        mask (np.ndarray): Mask over which to compute the coefficient.

    Returns:
        float: Mean value of the coefficient over the mask.
        0 if all the domain is masked.
    """
    if np.sum(mask) == 0:
        return 0
    return np.mean(bottom[mask] / top[mask])


def compute_coef_std_from_two_layers(
    top: np.ndarray,
    bottom: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute the standard deviation of the coefficient.

    Args:
        top (np.ndarray): Top layer.
        bottom (np.ndarray): Bottom layer.
        mask (np.ndarray): Mask over which to compute the std.

    Returns:
        float: std value of the coefficient over the mask.
        0 if all the domain is masked.
    """
    if np.sum(mask) == 0:
        return 0
    return np.std(bottom[mask] / top[mask])


def compute_coef_from_file(file: Path, field: str = "omega") -> float:
    """Compute the coefficient from a file.

    Args:
        file (Path): File to use.
        field (str, optional): Field to consider for the coefficient
        computation. Defaults to "omega".

    Returns:
        float: Coefficient value.
    """
    data = np.load(file)[field]
    top = data[0, 0]
    bottom = data[0, 1]
    mask = np.abs(top) > mean_threshold(top)
    return compute_coef_from_two_layers(top, bottom, mask)


def compute_coefficients(
    files: list[Path],
    field: str = "omega",
) -> np.ndarray:
    """Compute the coefficient over a list of fileS.

    Args:
        files (list[Path]): Files.
        field (str, optional): Field to consider for the coefficient
        computation. Defaults to "omega".

    Returns:
        np.ndarray: Coeeficient values for every file.
    """
    coefs = [compute_coef_from_file(file, field=field) for file in files]
    return np.array(coefs)


class Coefficient(metaclass=ABCMeta):
    """Base class for the coefficient."""

    _dt: float
    _step = 0
    _constant = False

    @property
    def isconstant(self) -> bool:
        """Whether the coefficient is constant."""
        return self._constant

    def __repr__(self) -> str:
        """Self representation.

        Returns:
            str: String value of the current parameter.
        """
        return self.at_step(self._step).__repr__()

    @abstractmethod
    def at_step(self, step: int) -> float:
        """Compute the coefficient at a given step.

        Args:
            step (int): Step at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """

    def at_time(self, time: float) -> float:
        """Compute the coefficient at a given time.

        Args:
            time (int): Time at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return self.at_step(time / self._dt)

    def reset_steps(self) -> None:
        """Reset steps numbers."""
        self._step = 0

    def next_value(self) -> float:
        """Next coefficient value.

        Automatically keep tracks of steps.

        Returns:
            float: coefficient value.
        """
        alpha = self.at_step(self._step)
        self._step += 1
        return alpha


class ConstantCoefficient(Coefficient):
    """Constant Coefficient."""

    _constant = True

    def __init__(self, value: float) -> None:
        """Instantiate the coefficient.

        Args:
            value (float): Coefficient constant value.
        """
        self._value = value

    def at_step(self, step: int) -> float:  # noqa: ARG002
        """Compute the coefficient at a given step.

        Args:
            step (int): Step at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return self._value

    def at_time(self, time: float) -> float:  # noqa: ARG002
        """Compute the coefficient at a given time.

        Args:
            time (int): Time at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return self._value


class ChangingCoefficient(Coefficient):
    """Changing coefficient."""

    sigma = 0.25
    _kernel_width = 30

    def __init__(
        self,
        coefficients: np.ndarray,
        steps: np.ndarray,
        dt: float,
    ) -> None:
        """Instantiate ChangingCoefficient.

        Args:
            coefficients (np.ndarray): Coefficients values.
            steps (np.ndarray): Steps.
            dt (float): Timestep (in seconds).
        """
        self._coefs = coefficients
        self._steps = steps
        self._dt = dt
        self._coef_interpolation = self._interpolate_coefs(
            self._steps,
            self._coefs,
        )

    @property
    def kernel_width(self) -> int:
        """Width of the smoothing kernel."""
        return self._kernel_width

    def _interpolate_coefs(
        self,
        steps: np.ndarray,
        coefficients: np.ndarray,
    ) -> interpolate.interp1d:
        """Interpolate coefficients using steps.

        Args:
            steps (np.ndarray): Steps.
            coefficients (np.ndarray): Coefficients.

        Returns:
            interpolate.interp1d: Interpolator.
        """
        smoothed_coefs = self._smooth(coefficients)
        return interpolate.interp1d(steps, y=smoothed_coefs)

    def _smooth(self, coefficients: np.ndarray) -> np.ndarray:
        """Smooth coefficients using a gaussian kernel.

        Args:
            coefficients (np.ndarray): Coefficients.

        Returns:
            np.ndarray: Smoothed coefficients.
        """
        kernel = self._generate_kernel()
        pad = (kernel.shape[0] - 1, kernel.shape[0] - 1)
        pad_coefs = np.pad(coefficients, pad, mode="edge")
        convolved = np.convolve(pad_coefs, kernel, mode="same")
        return convolved[pad[0] : -pad[1]]

    def _generate_kernel(self) -> np.ndarray:
        """Generate filtering kernel.

        Returns:
            np.ndarray: Filtering kernel.
        """
        x = np.linspace(-1, 1, int(self.kernel_width))
        kernel = (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(-np.power(x, 2) / (2 * self.sigma**2))
        )
        return kernel / np.sum(kernel)

    def at_step(self, step: int) -> float:
        """Compute the coefficient at a given step.

        Args:
            step (int): Step at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return float(self._coef_interpolation(step))

    def adjust_kernel_width(self, Ro: float, f0: float) -> None:  # noqa: N803
        """Adjust the kernle width based on physicial parameters.

        Args:
            Ro (float): Rossby Number.
            f0 (float): Coriolis Parameter

        Raises:
            ValueError: If steps have uneven spacing.
        """
        steps_spacing = self._steps[:-1][1:] - self._steps[:-1][:-1]
        if len(np.unique(steps_spacing)) != 1:
            msg = "Unable to retrieve steps spacing."
            raise ValueError(msg)
        returning_time = 1 / (Ro * f0)
        returning_steps = returning_time / self._dt
        self._kernel_width = returning_steps // steps_spacing[0] + 1
        self._coef_interpolation = self._interpolate_coefs(
            self._steps,
            self._coefs,
        )

    @classmethod
    def from_model_files(
        cls,
        files: list[Path],
        steps: np.ndarray,
        dt: float,
        field: str = "omega",
    ) -> Self:
        """Instantiate the coefficient from a list of files.

        Args:
            files (list[Path]): Files to use.
            steps (np.ndarray): Steps values.
            dt (float): Timesetp value (in seconds).
            field (str, optional): Field to consider. Defaults to "omega".

        Returns:
            Self: Coefficient.
        """
        return cls(
            compute_coefficients(files, field=field),
            steps=steps,
            dt=dt,
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        dt: float,
        coefs_field: str = "alpha",
        steps_fields: str = "steps",
    ) -> Self:
        """Instantiate the coefficient a file of values and steps.

        Args:
            file (Path): File to laod.
            dt (float): Timestep value (in seconds).
            coefs_field (str, optional): Field for coefs. Defaults to "alpha".
            steps_fields (str, optional): Field for steps. Defaults to "steps".

        Returns:
            Self: Coefficient.
        """
        data = np.load(file)
        return cls(
            data[coefs_field],
            data[steps_fields],
            dt=dt,
        )
