"""Model Data retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgsw import verbose
from qgsw.models.exceptions import InvalidSavingFileError

if TYPE_CHECKING:
    from pathlib import Path

    from qgsw.models.variables.base import (
        BoundDiagnosticVariable,
        PrognosticVariable,
    )
    from qgsw.models.variables.prognostic import (
        LayerDepthAnomaly,
        MeridionalVelocity,
        ZonalVelocity,
    )


class IO:
    """Input/Output manager."""

    def __init__(
        self,
        u: ZonalVelocity,
        v: MeridionalVelocity,
        h: LayerDepthAnomaly,
        *args: BoundDiagnosticVariable,
    ) -> None:
        """Instantiate the object.

        Args:
            u (ZonalVelocity): Zonal velocity.
            v (MeridionalVelocity): Meriodional velocity.
            h (LayerDepthAnomaly): Layer depth anomaly.
            *args (BoundDiagnosticVariable): Diagnostic variableS.
        """
        self._prog: list[PrognosticVariable] = [u, v, h]
        self._diag = set(args)

    def _raise_if_invalid_savefile(self, output_file: Path) -> None:
        """Raise an error if the saving file is invalid.

        Args:
            output_file (Path): Output file.

        Raises:
            InvalidSavingFileError: if the saving file extension is not .npz.
        """
        if output_file.suffix != ".npz":
            msg = "Variables are expected to be saved in an .npz file."
            raise InvalidSavingFileError(msg)

    def save(self, output_file: Path) -> None:
        """Save given variables.

        Args:
            output_file (Path): File to save value in (.npz).
        """
        self._raise_if_invalid_savefile(output_file=output_file)
        to_save = {
            var.name: var.get().cpu().numpy().astype("float32")
            for var in self._prog
        }
        to_save |= {
            var.name: var.get().cpu().numpy().astype("float32")
            for var in self._diag
        }
        np.savez(
            file=output_file,
            **to_save,
        )

        verbose.display(
            msg=f"Saved {', '.join(list(to_save.keys()))} to {output_file}.",
            trigger_level=1,
        )

    def add_diagnostic_vars(self, *args: BoundDiagnosticVariable) -> None:
        """Add a diagnostic varibale to track.

        Args:
            *args (BoundDiagnosticVariable): Diagnostic variable.
        """
        self._diag |= set(args)

    def print_step(self) -> str:
        """Printable informations of the prognostic variables.

        Returns:
            str: Informations to print.
        """
        snippets = []
        for var in self._prog:
            data = var.get()
            data_mean = data.mean().cpu().item()
            data_max = data.abs().max().cpu().item()
            snippets.append(
                f"{var.name}: mean: {data_mean:+.3E}, max: {data_max:+.3E}",
            )
        return " - ".join(snippets)

    def remove_diagnostic_vars(self) -> None:
        """Remove diagnostic variables tracking."""
        self._diag = set()
