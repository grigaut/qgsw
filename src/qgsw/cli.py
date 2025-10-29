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

    indices: list[int]
    comparison: int
    cycles: int
    prefix: str
    no_wind: bool = False

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
            nargs="+",
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
