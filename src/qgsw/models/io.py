"""Model Data retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qgsw import verbose
from qgsw.models.exceptions import InvalidSavingFileError

if TYPE_CHECKING:
    from pathlib import Path

    from qgsw.variables.base import (
        PrognosticVariable,
    )
    from qgsw.variables.prognostic import (
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
    ) -> None:
        """Instantiate the object.

        Args:
            u (ZonalVelocity): Zonal velocity.
            v (MeridionalVelocity): Meriodional velocity.
            h (LayerDepthAnomaly): Layer depth anomaly.
            *args (BoundDiagnosticVariable): Diagnostic variableS.
        """
        self._prog: list[PrognosticVariable] = [u, v, h]

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
        np.savez(
            file=output_file,
            **to_save,
        )

        verbose.display(
            msg=f"Saved {', '.join(list(to_save.keys()))} to {output_file}.",
            trigger_level=1,
        )

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
