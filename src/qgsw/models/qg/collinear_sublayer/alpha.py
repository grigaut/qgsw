"""Compute Collinearity Coefficient."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch

from qgsw.specs import DEVICE

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from qgsw.utils.gaussian_filtering import GaussianFilter1D
from qgsw.utils.type_switch import TypeSwitch

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from qgsw.configs.models import CollinearityCoefficientConfig


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

        self._filter = GaussianFilter1D(sigma=0.25, radius=30)

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
        "constant",
        ChangingCoefficient.get_type(),
    ]
    if coef_config.type not in possible_coefs:
        msg = (
            "Unrecognized perturbation type. "
            f"Possible values are {possible_coefs}"
        )
        raise KeyError(msg)

    if coef_config.type == "constant":
        coef = torch.tensor(
            [coef_config.value],
            dtype=torch.float64,
            device=DEVICE.get(),
        )
    if coef_config.type == ChangingCoefficient.get_type():
        msg = "To Implement."
        raise NotImplementedError(msg)
    return coef
