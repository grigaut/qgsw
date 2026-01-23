"""Command Line Interface."""

import argparse
import pathlib
from dataclasses import dataclass
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@dataclass
class ScriptArgs:
    """Script arguments."""

    config: Path
    verbose: int

    @classmethod
    def from_cli(cls) -> Self:
        """Instantiate script arguments from CLI.

        Returns:
            Self: ScriptArgs.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        cls._add_config(parser)
        cls._add_verbose(parser)
        return cls(**vars(parser.parse_args()))

    @classmethod
    def _add_config(cls, parser: argparse.ArgumentParser) -> None:
        """Add configuration to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "--config",
            required=True,
            type=pathlib.Path,
            help="Configuration File Path (from qgsw root level)",
        )

    @classmethod
    def _add_verbose(cls, parser: argparse.ArgumentParser) -> None:
        """Add verbose to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )


@dataclass
class ScriptArgsVA(ScriptArgs):
    """Script arguments."""

    indices: tuple[int, int, int, int]
    comparison: int
    cycles: int
    prefix: str
    no_wind: bool = False
    obs_track: bool = False

    @classmethod
    def from_cli(
        cls,
        *,
        comparison_default: int = 1,
        cycles_default: int = 3,
        prefix_default: str = "results",
    ) -> Self:
        """Instantiate script arguments from CLI.

        Args:
            comparison_default (int, optional): Default value
                for comparison interval. Defaults to 1.
            cycles_default (int, optional): Default value
                for number of cycles. Defaults to 3.
            prefix_default (str, optional): Default value for
                output file prefix. Defaults to "results".

        Returns:
            Self: ScriptArgsVA.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        cls._add_config(parser)
        cls._add_verbose(parser)
        cls._add_indices(parser)
        cls._add_comparison_interval(parser, comparison_default)
        cls._add_cycles(parser, cycles_default)
        cls._add_prefix(parser, prefix_default)
        cls._add_wind(parser)
        cls._add_obs_track(parser)
        return cls(**vars(parser.parse_args()))

    @classmethod
    def _add_indices(cls, parser: argparse.ArgumentParser) -> None:
        """Add indices to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "-i",
            "--indices",
            required=True,
            nargs=4,
            type=int,
            help="Indices (imin, imax, jmin, jmax), "
            "for example (64, 128, 128, 256).",
        )

    @classmethod
    def _add_comparison_interval(
        cls, parser: argparse.ArgumentParser, default: int
    ) -> None:
        """Add comparison interval to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the comparison interval.
        """
        parser.add_argument(
            "-c",
            "--comparison",
            type=int,
            default=default,
            help="Comparison interval value.",
        )

    @classmethod
    def _add_cycles(
        cls, parser: argparse.ArgumentParser, default: int
    ) -> None:
        """Add number of cycles to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the comparison interval.
        """
        parser.add_argument(
            "--cycles",
            type=int,
            default=default,
            help="Number of cycles.",
        )

    @classmethod
    def _add_prefix(
        cls, parser: argparse.ArgumentParser, default: int
    ) -> None:
        """Add output file prefix to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (int): Default value for for the prefix.
        """
        parser.add_argument(
            "-p",
            "--prefix",
            type=str,
            default=default,
            help="File saving prefix prefix.",
        )

    @classmethod
    def _add_wind(cls, parser: argparse.ArgumentParser) -> None:
        """Specify whether to use wind or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "--no-wind",
            action="store_true",
            help="Disable wind forcing.",
        )

    @classmethod
    def _add_obs_track(cls, parser: argparse.ArgumentParser) -> None:
        """Specify whether to use observation trackes or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "--obs-track",
            action="store_true",
            help="Enable observations tracks.",
        )

    def _build_suffix(self) -> list[str]:
        return [
            "_nowind" if self.no_wind else "",
            "_obstrack" if self.obs_track else "",
            f"_c{self.comparison}" if self.comparison != 1 else "",
        ]

    def complete_prefix(self) -> str:
        """Complete prefix with cli arguments info.

        Returns:
            str: Completed prefix.
        """
        return self.prefix + "".join(self._build_suffix())


@dataclass
class ScriptArgsVARegularized(ScriptArgsVA):
    """Script arguments."""

    no_reg: bool = False
    gamma: float = 1

    @classmethod
    def from_cli(
        cls,
        *,
        comparison_default: int = 1,
        cycles_default: int = 3,
        prefix_default: str = "results",
        gamma_default: float = 0,
    ) -> Self:
        """Instantiate script arguments from CLI.

        Args:
            comparison_default (int, optional): Default value
                for comparison interval. Defaults to 1.
            cycles_default (int, optional): Default value
                for number of cycles. Defaults to 3.
            prefix_default (str, optional): Default value for
                output file prefix. Defaults to "results".
            gamma_default (float): Default value for gamme. Defaults to 0.

        Returns:
            Self: ScriptArgsVA.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        cls._add_config(parser)
        cls._add_verbose(parser)
        cls._add_indices(parser)
        cls._add_comparison_interval(parser, comparison_default)
        cls._add_cycles(parser, cycles_default)
        cls._add_prefix(parser, prefix_default)
        cls._add_wind(parser)
        cls._add_reg(parser)
        cls._add_gamma(parser, gamma_default)
        cls._add_obs_track(parser)
        return cls(**vars(parser.parse_args()))

    @classmethod
    def _add_gamma(
        cls, parser: argparse.ArgumentParser, default: float
    ) -> None:
        """Specify whether to use regularization or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default (float): Default value.
        """
        parser.add_argument(
            "--gamma",
            type=float,
            default=default,
            help="Gamma value.",
        )

    @classmethod
    def _add_reg(cls, parser: argparse.ArgumentParser) -> None:
        """Specify whether to use regularization or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "--no-reg",
            action="store_true",
            help="Disable regularization.",
        )

    def _build_suffix(self) -> list[str]:
        if (g := self.gamma) != 0:
            gamma_str = str(g).rstrip("0").rstrip(".").replace(".", "_")
        else:
            gamma_str = "0"
        return [
            "_noreg" if self.no_reg else "",
            f"_gamma{gamma_str}"
            if (not self.no_reg and self.gamma != 1)
            else "",
            *super()._build_suffix(),
        ]


@dataclass
class ScriptArgsVAModified(ScriptArgsVARegularized):
    """Script arguments."""

    no_alpha: bool = False

    @classmethod
    def from_cli(
        cls,
        *,
        comparison_default: int = 1,
        cycles_default: int = 3,
        prefix_default: str = "results",
        gamma_default: float = 0,
    ) -> Self:
        """Instantiate script arguments from CLI.

        Args:
            comparison_default (int, optional): Default value
                for comparison interval. Defaults to 1.
            cycles_default (int, optional): Default value
                for number of cycles. Defaults to 3.
            prefix_default (str, optional): Default value for
                output file prefix. Defaults to "results".
            gamma_default (float): Default value for gamme. Defaults to 0.

        Returns:
            Self: ScriptArgsVA.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        cls._add_config(parser)
        cls._add_verbose(parser)
        cls._add_indices(parser)
        cls._add_comparison_interval(parser, comparison_default)
        cls._add_cycles(parser, cycles_default)
        cls._add_prefix(parser, prefix_default)
        cls._add_wind(parser)
        cls._add_reg(parser)
        cls._add_gamma(parser, gamma_default)
        cls._add_alpha(parser)
        cls._add_obs_track(parser)
        return cls(**vars(parser.parse_args()))

    @classmethod
    def _add_alpha(cls, parser: argparse.ArgumentParser) -> None:
        """Specify whether to use alpha or not.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "--no-alpha",
            action="store_true",
            help="Disable regularization.",
        )

    def _build_suffix(self) -> list[str]:
        return [
            "_noalpha" if self.no_alpha else "",
            *super()._build_suffix(),
        ]
