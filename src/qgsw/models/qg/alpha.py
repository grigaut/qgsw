"""Compute Collinearity Coefficient."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from typing_extensions import Self

from qgsw.run_summary import RunSummary
from qgsw.utils.gaussian_filtering import GaussianFilter1D
from qgsw.utils.sorting import sort_files
from qgsw.utils.type_switch import TypeSwitch

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from qgsw.configs.alpha import CollinearityCoefficientConfig
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


def extract_coefficients_from_run(
    folder: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the coeffcient values from a run folder.

    Args:
        folder (Path): Run folder.

    Returns:
        tuple[np.ndarray, np.ndarray]: Times, coefficient values.
    """
    run = RunSummary.from_file(folder.joinpath("_summary.toml"))
    dt = run.configuration.simulation.dt
    steps, files = sort_files(
        folder.rglob("*.npz"),
        prefix=run.configuration.model.prefix,
        suffix=".npz",
    )
    coefficients = compute_coefficients(files, field="omega")
    return np.array(steps) * dt, coefficients


def save_coefficients(
    file: Path,
    times: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    """Save coeffciients in a .npz file.

    Args:
        file (Path): File to save in.
        times (np.ndarray): Steps.
        coefficients (np.ndarray): Coefficients.
        dt (float): Dt.
    """
    np.savez(file, times=times, alpha=coefficients)


class Coefficient(TypeSwitch, metaclass=ABCMeta):
    """Base class for the coefficient."""

    _time = 0
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
        return f"{self.type}: {self.at_current_time().__repr__()}"

    @abstractmethod
    def at_time(self, time: float) -> float:
        """Compute the coefficient at a given time.

        Args:
            time (int): Time at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return self.at_step(time / self._dt)

    def reset_time(self) -> None:
        """Reset time."""
        self._time = 0

    def next_time(self, dt: float) -> float:
        """Next coefficient value.

        Automatically keep tracks of steps.

        Returns:
            float: coefficient value.
        """
        alpha = self.at_time(self._time)
        self.to_time(self._time + dt)
        return alpha

    def at_current_time(self) -> float:
        """Value at current time.

        Returns:
            float: Coefficient value
        """
        return self.at_time(self._time)

    def to_time(self, time: float) -> None:
        """Change registered time value.

        Args:
            time (float): Time to move to.
        """
        self._time = time


class ConstantCoefficient(Coefficient):
    """Constant Coefficient."""

    _type = "constant"
    _constant = True

    def __init__(self, value: float) -> None:
        """Instantiate the coefficient.

        Args:
            value (float): Coefficient constant value.
        """
        self._value = value

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

    _type = "changing"

    def __init__(
        self,
        coefficients: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Instantiate ChangingCoefficient.

        Args:
            coefficients (np.ndarray): Coefficients values.
            times (np.ndarray): Times.
            dt (float): Timestep (in seconds).
        """
        self._coefs = coefficients
        self._times = times

        self._filter = GaussianFilter1D(sigma=0.25, kernel_width=30)

        self._coef_interpolation = self._interpolate_coefs(
            self._times,
            self._coefs,
        )

    @property
    def gaussian_filter(self) -> GaussianFilter1D:
        """Gaussian Filter."""
        return self._filter

    def _interpolate_coefs(
        self,
        times: np.ndarray,
        coefficients: np.ndarray,
    ) -> interpolate.interp1d:
        """Interpolate coefficients using times.

        Args:
            times (np.ndarray): Steps.
            coefficients (np.ndarray): Coefficients.

        Returns:
            interpolate.interp1d: Interpolator.
        """
        smoothed_coefs = self.gaussian_filter.smooth(coefficients)
        return interpolate.interp1d(times, y=smoothed_coefs)

    def at_time(self, time: int) -> float:
        """Compute the coefficient at a given time.

        Args:
            time (int): Step at which to compute the coefficient.

        Returns:
            float: Coefficient value.
        """
        return float(self._coef_interpolation(time))

    def adjust_kernel_width(self, Ro: float, f0: float) -> None:  # noqa: N803
        """Adjust the kernle width based on physicial parameters.

        Args:
            Ro (float): Rossby Number.
            f0 (float): Coriolis Parameter

        Raises:
            ValueError: If steps have uneven spacing.
        """
        dt = self._times[:-1][1:] - self._times[:-1][:-1]
        if len(np.unique(dt)) != 1:
            msg = "Unable to retrieve steps spacing."
            raise ValueError(msg)
        returning_time = 1 / (Ro * f0)
        self.gaussian_filter.kernel_width = int(returning_time // dt[0]) + 1
        self._coef_interpolation = self._interpolate_coefs(
            self._times,
            self._coefs,
        )

    def show(self, current: bool = True) -> None:  # noqa: FBT001, FBT002
        """Show the coefficient evolution.

        Args:
            current (bool, optional): Whether to show current time or not.
            Defaults to True.
        """
        ax: Axes
        _, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            self._times,
            self._coefs,
            c="orange",
            alpha=0.75,
            marker=".",
            label="Reference Values.",
        )
        ax.plot(
            self._times,
            self.gaussian_filter.smooth(self._coefs),
            c="blue",
            label="Smoothed Values.",
        )
        if current:
            ax.scatter(
                self._time,
                self.at_time(self._time),
                c="red",
                marker="o",
                label="Current Time.",
            )
        plt.legend()
        plt.show()

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
            times=steps * dt,
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        coefs_field: str = "alpha",
        times_field: str = "times",
    ) -> Self:
        """Instantiate the coefficient a file of values and times.

        Args:
            file (Path): File to laod.
            coefs_field (str, optional): Field for coefs. Defaults to "alpha".
            times_field (str, optional): Field for times. Defaults to "times".
            dt_field (str, optional): Field for dt. Defaults to "dt".

        Returns:
            Self: Coefficient.
        """
        data = np.load(file)
        return cls(
            data[coefs_field],
            data[times_field],
        )

    @classmethod
    def from_run(
        cls,
        folder: Path,
    ) -> Self:
        """Instantiate the coefficients from a run folder.

        Args:
            folder (Path): Folder of the run output.

        Returns:
            Self: Coefficient.
        """
        times, coefficients = extract_coefficients_from_run(folder)
        return cls(
            coefficients,
            times,
        )


def coefficient_from_config(
    coef_config: CollinearityCoefficientConfig,
) -> Coefficient:
    """Create Coefficient from configuration.

    Args:
        coef_config (CollinearityCoefficientConfig): Coefficient Configuration.

    Raises:
        KeyError: If the coeffciient type is not recognized/

    Returns:
        Coefficient: Coefficient.
    """
    possible_coefs = [
        ConstantCoefficient.get_type(),
        ChangingCoefficient.get_type(),
    ]
    if coef_config.type not in possible_coefs:
        msg = (
            "Unrecognized perturbation type. "
            f"Possible values are {possible_coefs}"
        )
        raise KeyError(msg)

    if coef_config.type == ConstantCoefficient.get_type():
        coef = ConstantCoefficient(value=coef_config.value)
    if coef_config.type == ChangingCoefficient.get_type():
        coef = ChangingCoefficient.from_file(coef_config.source_file)
    return coef
