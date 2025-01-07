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

        Args:
            default_config (str): Default configuration path.

        Returns:
            Self: ScriptArgs.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        parser.add_argument(
            "--config",
            required=True,
            type=pathlib.Path,
            help="Configuration File Path (from qgsw root level)",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )
        return cls(**vars(parser.parse_args()))
