"""Tools for outputs."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import pytz
import torch

from qgsw.logging.core import getLogger

logger = getLogger(__name__)


def create_output_folder(
    root_output_folder: Path | str,
    prefix: str | None = None,
    *,
    mkdir: bool = True,
    with_uuid: bool = True,
) -> Path:
    """Create output folder.

    Args:
        root_output_folder (Path | str): Root output folder path.
        prefix (str | None, optional): Any prefix to include. Defaults to None.
        mkdir (bool, optional): Whether to make directory of not.
            Defaults to True.
        with_uuid (bool, optional): Wehtehr to include uuid or not.
            Defaults to True.

    Returns:
        Path: Output folder path.
    """
    sep = "-"
    root_output_folder = Path(root_output_folder).absolute()

    prefix_str = prefix + sep if prefix is not None else ""
    datetime_str = datetime.now(pytz.utc).strftime("%Y%m%d_%H%M%S")

    folder = (prefix_str + datetime_str) + (sep + str(uuid4())[:8]) * with_uuid

    result_folder = root_output_folder / folder

    if mkdir:
        result_folder.mkdir(parents=True)

    return result_folder


def default_serializer(obj: Any) -> str:  # noqa: ANN401
    """Serializer for json.dump."""
    if isinstance(obj, Path):
        return str(obj)
    msg = f"Type {type(obj)} non serializable"
    raise TypeError(msg)


class OutputFile:
    """Output file handler."""

    _load: Callable[[], Any]
    _save: Callable[[Any], None]

    def __init__(self, filename: Path) -> None:
        """Instantiate OutputFile.

        Args:
            filename (Path): Filename.

        Raises:
            ValueError: If the file suffix is not valid.
        """
        self._file = filename
        match filename.suffix[1:]:
            case "json":
                self._load = self._load_json
                self._save = self._save_json
            case "pt":
                self._load = self._load_pt
                self._save = self._save_pt
            case _:
                msg = "Unsupported ouput file type."
                raise ValueError(msg)

    @property
    def file(self) -> Path:
        """Filename."""
        return self._file

    def _load_json(self) -> dict:
        with self.file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pt(self) -> dict:
        return torch.load(self._file, weights_only=False)

    def load(self) -> Any:  # noqa: ANN401
        """Load data from file.

        Returns:
            Any: Loaded data.
        """
        return self._load()

    def save(self, obj: Any) -> None:  # noqa: ANN401
        """Save data to file.

        Args:
            obj (Any): Data to save.
        """
        self._save(obj)
        msg = f"Saved to {self._file}."
        logger.detail(msg)

    def _save_json(self, obj: Any) -> None:  # noqa: ANN401
        with self.file.open("w", encoding="utf-8") as f:
            json.dump(obj, f, default=default_serializer, indent=4)

    def _save_pt(self, obj: Any) -> None:  # noqa: ANN401
        torch.save(obj, self._file)


class OutputFolder(NamedTuple):
    """Output folder."""

    folder: Path
    params: OutputFile
    results: OutputFile
    basis: OutputFile
    optim: OutputFile


def build_output_folder(
    output_folder: Path | str,
    params: str = "params",
    results: str = "results",
    basis: str = "basis",
    optim: str = "optim",
) -> OutputFolder:
    """Build output folder.

    Args:
        output_folder (Path|str): Path to output folder
        params (str, optional): Name of the params file. Defaults to "params".
        results (str, optional): Name of the results file.
            Defaults to "results".
        basis (str, optional): Name of the basis file. Defaults to "basis".
        optim (str, optional): Name of the optim file. Defaults to "optim".

    Returns:
        OutputFolder: Output
    """
    output_folder = Path(output_folder)
    if not output_folder.exists():
        msg = "The output folder does not exist."
        raise ValueError(msg)
    paramsf = OutputFile(output_folder / f"{params}.json")
    resultsf = OutputFile(output_folder / f"{results}.pt")
    basisf = OutputFile(output_folder / f"{basis}.pt")
    optimf = OutputFile(output_folder / f"{optim}.json")
    return OutputFolder(
        folder=output_folder,
        params=paramsf,
        results=resultsf,
        basis=basisf,
        optim=optimf,
    )
