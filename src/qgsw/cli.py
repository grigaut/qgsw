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
        cls._add_indices(parser)
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
