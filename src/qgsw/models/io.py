"""Model Data retriever."""

# ruff: noqa: A005

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from qgsw import verbose
from qgsw.exceptions import InvalidSavingFileError
from qgsw.fields.variables.prognostic import Time

if TYPE_CHECKING:
    from qgsw.fields.variables.base import (
        BoundDiagnosticVariable,
        PrognosticVariable,
    )


class IO:
    """Input/Output manager."""

    def __init__(
        self,
        *args: PrognosticVariable | BoundDiagnosticVariable,
        **kwargs: PrognosticVariable | BoundDiagnosticVariable,
    ) -> None:
        """Instantiate the object.

        Args:
            *args (BoundDiagnosticVariable): Prognostic variables.
            **kwargs (BoundDiagnosticVariable): Prognostic variables.
        """
        self._vars: list[PrognosticVariable | BoundDiagnosticVariable] = list(
            args,
        ) + list(
            kwargs.values(),
        )

    @property
    def vars(self) -> list[PrognosticVariable]:
        """Prognostic variables."""
        return self._vars

    def _raise_if_invalid_savefile(self, output_file: Path) -> None:
        """Raise an error if the saving file is invalid.

        Args:
            output_file (Path): Output file.

        Raises:
            InvalidSavingFileError: if the saving file extension is not .pt.
        """
        if output_file.suffix != ".pt":
            msg = "Variables are expected to be saved in an .pt file."
            raise InvalidSavingFileError(msg)

    def save(self, output_file: Path) -> None:
        """Save given variables.

        Args:
            output_file (Path): File to save value in (.pt).
        """
        output_file = Path(output_file)
        self._raise_if_invalid_savefile(output_file=output_file)
        to_save = {
            var.name: var.get().to(torch.float32).cpu() for var in self._vars
        }
        torch.save(to_save, f=output_file)

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
        for var in self._vars:
            if var.name == Time.get_name():
                snippets.append(
                    f"{var.name} [{var.unit.value}]: {var.get().cpu().item()}",
                )
                continue
            data = var.get()
            data_mean = data.mean().cpu().item()
            data_max = data.abs().max().cpu().item()
            snippets.append(
                f"{var.name} [{var.unit.value}]: mean: {data_mean:+.3E},"
                f" max: {data_max:+.3E}",
            )
        return " - ".join(snippets)
