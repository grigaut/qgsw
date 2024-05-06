"""Generate run summaries."""

from importlib.metadata import version
from pathlib import Path
from typing import Any

import toml
from typing_extensions import Self

from qgsw.configs import Configuration


class RunSummary:
    """Run summary."""

    summary_section: str = "run-summary"

    def __init__(self, run_params: dict[str, Any]) -> None:
        """Instantiate run summary.

        Args:
            run_params (dict[str, Any]): Run parameters.
        """
        if self.summary_section in run_params:
            self._has_summary = True
            self._summary = run_params
        else:
            self._has_summary = False
            self._summary = {"configuration": run_params}
            self._summary[self.summary_section] = {}
            self._summary["qgsw-version"] = version("qgsw")
        self._files: list[Path] = []
        self._config = Configuration(self._summary["configuration"])

    @property
    def configuration(self) -> Configuration:
        """Run Configuration."""
        return self._config

    def raise_if_not_writable(self) -> None:
        """Raise an error is the configuration ins not writable.

        Raises:
            ValueError: If the summary section existed at instanciation.
        """
        if self._has_summary:
            msg = "Summary section already existed, impossible to override."
            raise ValueError(msg)

    def register_steps(self, *, t_end: float, dt: float, n_steps: int) -> None:
        """Register steps parameters.

        Args:
            t_end (float): Total simulation time in seconds.
            dt (float): Timestep in seconds.
            n_steps (int): Number of steps.
        """
        self.raise_if_not_writable()
        self._summary[self.summary_section]["total_duration"] = t_end
        self._summary[self.summary_section]["dt"] = dt
        self._summary[self.summary_section]["n_steps"] = n_steps
        if self._files:
            self.update()

    def register_start(self) -> None:
        """Register the start of the simulation."""
        self.raise_if_not_writable()
        self._summary[self.summary_section]["finished_run"] = False
        if self._files:
            self.update()

    def register_step(self, step: int) -> None:
        """Register a simulation step.

        Args:
            step (int): Number of the step.
        """
        self.raise_if_not_writable()
        self._summary[self.summary_section]["last_registered_step"] = step
        if self._files:
            self.update()

    def register_end(self) -> None:
        """Register the end of the simulation."""
        self.raise_if_not_writable()
        if self.summary_section not in self._summary:
            self._summary[self.summary_section] = {}
        self._summary[self.summary_section]["finished_run"] = True
        if self._files:
            self.update()

    def to_file(self, file: Path) -> None:
        """Save the summary into a file.

        Multiple calls will add as many files as required.

        Args:
            file (Path): File to save in.

        Raises:
            ValueError: If the file is not a TOML file.
        """
        self.raise_if_not_writable()
        if file.suffix != ".toml":
            msg = "Summary can only be saved into a .toml file."
            raise ValueError(msg)
        toml.dump(self._summary, file.open("w"))
        if file not in self._files:
            self._files.append(file)

    def update(self) -> None:
        """Update the saved files.

        Raises:
            ValueError: If no files are registered.
        """
        self.raise_if_not_writable()
        if not self._files:
            msg = "Impossible to update the summary, use .to_file first."
            raise ValueError(msg)
        if not self._files:
            msg = "Impossible to update the summary, use .to_file first."
            raise ValueError(msg)
        for file in self._files:
            toml.dump(self._summary, file.open("w"))

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """Create the summary from a TOML file.

        Args:
            file (Path): File to create from.

        Returns:
            Self: Summary.
        """
        return cls(run_params=toml.load(file))

    @classmethod
    def from_configuration(cls, configuration: Configuration) -> Self:
        """CReate the summary from a configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Summary.
        """
        return cls(run_params=configuration.params)
