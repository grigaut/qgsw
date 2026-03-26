"""Command Line Interface."""

from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ScriptsArgsParser:
    """Scripts arguments parser."""

    retrieved = False

    has_config = False
    has_verbose = False
    has_indices = False
    has_comparison = False
    has_cycles = False
    has_prefix = False
    has_no_wind = False
    has_obs_track = False
    has_optim = False
    has_separation = False
    has_season = False
    has_gamma = False
    has_no_reg = False
    has_no_alpha = False

    @property
    def config(self) -> Path:
        """Configuration."""
        self._check_attr(self.has_config, "Configuration not tracked.")
        return self.namespace.config

    @property
    def verbose(self) -> int:
        """Verbose level."""
        self._check_attr(self.has_verbose, "Verbose not tracked.")
        return self.namespace.verbose

    @property
    def indices(self) -> tuple[int, int, int, int]:
        """Indices."""
        self._check_attr(self.has_indices, "Indices not tracked.")
        return self.namespace.indices

    @property
    def comparison(self) -> int:
        """Comparison interval."""
        self._check_attr(
            self.has_comparison,
            "Comparison interval not tracked.",
        )
        return self.namespace.comparison

    @property
    def cycles(self) -> int:
        """Number of cycles."""
        self._check_attr(
            self.has_cycles,
            "Cycles not tracked.",
        )
        return self.namespace.cycles

    @property
    def prefix(self) -> str:
        """Prefix."""
        self._check_attr(
            self.has_prefix,
            "Prefix not tracked.",
        )
        return self.namespace.prefix

    @property
    def no_wind(self) -> bool:
        """No wind."""
        self._check_attr(
            self.has_no_wind,
            "Wind not tracked.",
        )
        return self.namespace.no_wind

    @property
    def obs_track(self) -> bool:
        """Observation tracks."""
        self._check_attr(
            self.has_obs_track,
            "Observation tracks not tracked.",
        )
        return self.namespace.obs_track

    @property
    def optim(self) -> int:
        """Max optimization steps."""
        self._check_attr(
            self.has_optim,
            "Max optimization steps tracks not tracked.",
        )
        return self.namespace.optim

    @property
    def separation(self) -> int:
        """Separation steps."""
        self._check_attr(
            self.has_separation,
            "Separation steps not tracked.",
        )
        return self.namespace.separation

    @property
    def season(self) -> str:
        """Seasons."""
        self._check_attr(
            self.has_season,
            "Seasons not tracked.",
        )
        return self.namespace.season

    @property
    def gamma(self) -> float:
        """Gamma."""
        self._check_attr(
            self.has_gamma,
            "Gamma not tracked.",
        )
        return self.namespace.gamma

    @property
    def no_reg(self) -> bool:
        """No regularization."""
        self._check_attr(
            self.has_no_reg,
            "Regularization not tracked.",
        )
        return self.namespace.no_reg

    @property
    def no_alpha(self) -> bool:
        """No alpha."""
        self._check_attr(
            self.has_no_alpha,
            "Alpha not tracked.",
        )
        return self.namespace.no_alpha

    def __init__(self) -> None:
        """Instantiate parser."""
        self.parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        self.namespace = argparse.Namespace()

    def _check_unretrieved(self) -> None:
        if self.retrieved:
            msg = "Arguments already retrieved yet."
            raise ValueError(msg)

    def _check_attr(self, has_attr: bool, msg: str) -> None:  # noqa: FBT001
        if not self.retrieved:
            msg = "Argument not retrieved yet, use ScriptArgs.retrieve()."
            raise ValueError(msg)
        if has_attr:
            return
        raise AttributeError(msg)

    def add_config(self) -> None:
        """Add configuration to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--config",
            required=True,
            type=pathlib.Path,
            help="Configuration File Path (from qgsw root level)",
        )
        self.has_config = True

    def add_verbose(self) -> None:
        """Add verbose to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )
        self.has_verbose = True

    def add_indices(self) -> None:
        """Add indices to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-i",
            "--indices",
            required=False,
            nargs=4,
            type=int,
            help="Indices (imin, imax, jmin, jmax), "
            "for example (64, 128, 128, 256).",
        )
        self.has_indices = True

    def add_comparison_interval(self, default: int = 1) -> None:
        """Add comparison interval to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int, optional): Default value for for the comparison
                interval. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-c",
            "--comparison",
            type=int,
            default=default,
            help="Comparison interval value.",
        )
        self.has_comparison = True

    def add_cycles(self, default: int = 3) -> None:
        """Add number of cycles to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the comparison interval.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--cycles",
            type=int,
            default=default,
            help="Number of cycles.",
        )
        self.has_cycles = True

    def add_prefix(self, default: str = "results") -> None:
        """Add output file prefix to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the prefix.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "-p",
            "--prefix",
            type=str,
            default=default,
            help="File saving prefix prefix.",
        )
        self.has_prefix = True

    def add_wind(self) -> None:
        """Specify whether to use wind or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-wind",
            action="store_true",
            help="Disable wind forcing.",
        )
        self.has_no_wind = True

    def add_obs_track(self) -> None:
        """Specify whether to use observation trackes or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--obs-track",
            action="store_true",
            help="Enable observations tracks.",
        )
        self.has_obs_track = True

    def add_optim_max_step(self, default: int = 200) -> None:
        """Specify number of optimization steps to perform.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--optim",
            "-o",
            type=int,
            default=default,
            help="Max optimization steps.",
        )
        self.has_optim = True

    def add_separation(self, default: int = 0) -> None:
        """Specify number of step to perform to separate cycles.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int, optional): Default value. Defaults to 0.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--separation",
            "-s",
            type=int,
            default=default,
            help="Number of step to separate cycles.",
        )
        self.has_separation = True

    def add_season(self, default: str = "summer") -> None:
        """Specify season.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (str | None): Default season. Defaults to "summer".
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--season",
            type=str,
            default=default,
            choices=["summer", "autumn", "winter", "spring"],
            help="Season.",
        )
        self.has_season = True

    def add_gamma(self, default: float = 1) -> None:
        """Specify whether to use regularization or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float, optional): Default value. Defaults to 1.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=default,
            help="Gamma value.",
        )
        self.has_gamma = True

    def add_no_reg(self) -> None:
        """Specify whether to use regularization or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-reg",
            action="store_true",
            help="Disable regularization.",
        )
        self.has_no_reg = True

    def add_regularization(self, gamma_default: float = 1) -> None:
        """Add regularization."""
        self.add_gamma(gamma_default)
        self.add_no_reg()

    def add_alpha(self) -> None:
        """Specify whether to use alpha or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        self._check_unretrieved()
        self.parser.add_argument(
            "--no-alpha",
            action="store_true",
            help="Disable regularization.",
        )
        self.has_no_alpha = True

    def retrieve(self) -> None:
        """Retrieve arguments."""
        self.parser.parse_args(namespace=self.namespace)
        self.retrieved = True

    def _build_suffix(self) -> None:
        """Build full prefix."""
        if self.has_gamma and (g := self.gamma) != 0:
            gamma_str = str(g).rstrip("0").rstrip(".").replace(".", "_")
        else:
            gamma_str = "0"
        gamma_suffix = (
            f"_gamma{gamma_str}"
            if (
                (self.has_no_reg and not self.no_reg)
                and (self.has_gamma and self.gamma != 1)
            )
            else ""
        )
        return [
            "_noalpha" if (self.has_no_alpha and self.no_alpha) else "",
            "_noreg" if (self.has_no_reg and self.no_reg) else "",
            gamma_suffix,
            "_nowind" if (self.has_no_wind and self.no_wind) else "",
            "_obstrack" if (self.has_obs_track and self.obs_track) else "",
            f"_c{self.comparison}"
            if (self.has_obs_track and self.obs_track)
            and (self.has_comparison and self.comparison != 1)
            else "",
            f"_o{self.optim}"
            if (self.has_optim and self.optim != 200)
            else "",
            f"_s{self.separation}"
            if (self.has_separation and self.separation != 0)
            else "",
            f"_{self.season}"
            if (self.has_season and self.season is not None)
            else "",
        ]

    def complete_prefix(self) -> str:
        """Complete prefix with cli arguments info.

        Returns:
            str: Completed prefix.
        """
        return self.prefix + "".join(self._build_suffix())

    @classmethod
    def va_setup(cls, prefix_default: str, cycles_default: int = 3) -> Self:
        """Pre-setup parser for variationla assimilaiton scripts."""
        obj = cls()
        obj.add_config()
        obj.add_verbose()
        obj.add_prefix(default=prefix_default)
        obj.add_comparison_interval(default=1)
        obj.add_wind()
        obj.add_obs_track()
        obj.add_cycles(default=cycles_default)
        obj.add_optim_max_step(default=200)
        obj.add_separation(default=0)
        return obj
