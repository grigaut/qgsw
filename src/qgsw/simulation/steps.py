"""Simulation steps."""

import itertools
from collections.abc import Iterator
from math import ceil, floor


class Steps:
    """Simulation steps.

    Steps object will always overshoot rather than undershoot number of steps.
    """

    __slots__ = ("_dt", "_n_tot", "_t_end", "_t_start")

    def __init__(self, t_start: float, t_end: float, dt: float) -> None:
        """Instantiate the steps.

        Args:
            t_start (float): Simulation starting time.
            t_end (float): Simulation ending time.
            dt (float): Simulation timestep (in seconds).

        Raises:
            ValueError: If the timestep is greater than the simulation time.
        """
        if (t_end != t_start) and (dt > (t_end - t_start)):
            msg = "Timestep must be lower than t_start - t_end."
            raise ValueError(msg)
        self._dt = dt
        self._t_start = t_start
        self._t_end = t_end
        # Always overshoot number of steps
        self._n_tot = self._compute_n_tot()

    def __repr__(self) -> str:
        """String representation for steps.

        Returns:
            str: String representation.
        """
        return (
            f"Perform {self._n_tot} steps of {self._dt} s "
            f"for a total of {self._t_end - self._t_start} s "
            f"({self._t_end} - {self._t_start})."
        )

    def _compute_n_tot(self) -> int:
        """Compute total number of steps.

        if (end-start) % dt <  dt / 2 -> ⌊(end - start) / dt⌋
        if (end-start) % dt >= dt / 2 -> ⌈(end - start) / dt⌉

        Returns:
            int: Total number of steps.
        """
        t_span = self._t_end - self._t_start
        steps_float = t_span / self._dt
        return (
            floor(steps_float)
            if t_span % self._dt < (self._dt / 2)
            else ceil(steps_float)
        )

    @property
    def n_tot(self) -> int:
        """Simulation's total number of steps."""
        return self._n_tot

    def simulation_steps(self) -> range:
        """Simulation steps.

        Returns:
            range: Steps.
        """
        return range(self.n_tot)

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
