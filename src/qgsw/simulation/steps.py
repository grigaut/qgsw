"""Simulation steps."""

import itertools
from collections.abc import Iterator
from math import ceil


class Steps:
    """Simulation steps."""

    def __init__(self, t_end: float, dt: float) -> None:
        """Instantiate the steps.

        Args:
            t_end (float): Simulation ending time.
            dt (float): SImulation timestep (in seconds).

        Raises:
            ValueError: If the timestep is greater than the simulation time.
        """
        if dt > t_end:
            msg = "Timestep must be lower than ending time."
            raise ValueError(msg)
        self._dt = dt
        self._t_end = t_end
        self._n_tot = ceil(t_end / dt)

    @property
    def n_tot(self) -> int:
        """Simulation's total number of steps."""
        return self._n_tot

    def simulation_steps(self) -> range:
        """Simulation steps.

        Returns:
            range: Steps.
        """
        return range(self.n_tot + 1)

    def steps_from_interval(self, interval: float) -> Iterator:
        """Select steps with a given interval.

        Args:
            interval (float): Interval between steps (seconds).

        Yields:
            Iterator: Boolean iterator with the same length as the one returned
            by `simulation_steps`, True when the corresponding step matches the
            interval, always ends with True.
        """
        steps = self.simulation_steps()[:-1]
        steps_filter = ((n * self._dt % interval) < self._dt for n in steps)
        return itertools.chain(steps_filter, [True])

    def steps_from_total(self, total: int) -> Iterator:
        """Select a given number of steps.

        Args:
            total (int): Number of steps to select.

        Yields:
            Iterator: Boolean iterator with the same length as the one returned
            by `simulation_steps`, True when the corresponding step is selected
            , always ends with True. May include one additional step.
        """
        return self.steps_from_interval(self.n_tot * self._dt / total)
