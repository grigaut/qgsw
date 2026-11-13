"""Tools for outputs."""

import json
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import pytz
import torch

from qgsw.logging.core import getLogger
from qgsw.logging.utils import box

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

    def __init__(self, filename: Path, *, dry_run: bool = False) -> None:
        """Instantiate OutputFile.

        Args:
            filename (Path): Filename.
            dry_run (bool, optional): NOthing will be saved if True.
                Defaults to False.

        Raises:
            ValueError: If the file suffix is not valid.
        """
        self._file = filename
        self.dry_run = dry_run
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
        if not self.dry_run:
            self._save(obj)
            msg = f"Saved to {self._file}."
            logger.detail(msg)
        else:
            msg = f"[dry run] Saved to {self._file}."
            logger.detail(msg)

    def _save_json(self, obj: Any) -> None:  # noqa: ANN401
        with self.file.open("w", encoding="utf-8") as f:
            json.dump(obj, f, default=default_serializer, indent=4)

    def _save_pt(self, obj: Any) -> None:  # noqa: ANN401
        torch.save(obj, self._file)


class OutputFolder(NamedTuple):
    """Output folder."""

    folder: Path
    config: Path
    params: OutputFile
    results: OutputFile
    basis: OutputFile
    optim: OutputFile
    scores: OutputFile

    @property
    def dry_run(self) -> bool:
        """Whether to perform dry run or not."""
        return (
            self.params.dry_run
            and self.results.dry_run
            and self.basis.dry_run
            and self.optim.dry_run
            and self.scores.dry_run
        )

    def get_repr_parts(self) -> list[str]:
        """Get representation parts."""
        return (
            [f"Folder : {self.folder}"]
            + [
                f"\t├── {s.capitalize()}: {getattr(self, s).file}"
                for s in [
                    "params",
                    "results",
                    "basis",
                    "optim",
                    "scores",
                ]
            ]
            + [f"\t└── Config: {self.config}"]
        )

    def __repr__(self) -> str:
        """String representation."""
        return "\n".join(self.get_repr_parts())

    def files_exist(self) -> bool:
        """Check that all file exist."""
        pexists = self.params.file.exists()
        rexists = self.results.file.exists()
        bexists = self.basis.file.exists()
        oexists = self.optim.file.exists()
        sexists = self.scores.file.exists()
        return pexists and rexists and bexists and oexists and sexists

    def copy_config(self, config_path: Path) -> None:
        """Copy config file."""
        if self.dry_run:
            return
        shutil.copy2(config_path, self.config)


def build_output_folder(
    output_folder: Path | str,
    config: str = "config",
    params: str = "params",
    results: str = "results",
    basis: str = "basis",
    optim: str = "optim",
    scores: str = "scores",
    *,
    dry_run: bool = False,
) -> OutputFolder:
    """Build output folder.

    Args:
        output_folder (Path|str): Path to output folder
        config (str): Name of the config file. Defaults to "config".
        params (str, optional): Name of the params file. Defaults to "params".
        results (str, optional): Name of the results file.
            Defaults to "results".
        basis (str, optional): Name of the basis file. Defaults to "basis".
        optim (str, optional): Name of the optim file. Defaults to "optim".
        scores (str, optional): Name of the scores file. Defaults to "scores".
        dry_run (bool, optional): NOthing will be saved if True.
                Defaults to False.

    Returns:
        OutputFolder: Output
    """
    output_folder = Path(output_folder)
    if not output_folder.exists() and not dry_run:
        msg = "The output folder does not exist."
        raise FileNotFoundError(msg)
    configf = output_folder / f"{config}.toml"
    paramsf = OutputFile(output_folder / f"{params}.json", dry_run=dry_run)
    resultsf = OutputFile(output_folder / f"{results}.pt", dry_run=dry_run)
    basisf = OutputFile(output_folder / f"{basis}.pt", dry_run=dry_run)
    optimf = OutputFile(output_folder / f"{optim}.json", dry_run=dry_run)
    scoresf = OutputFile(output_folder / f"{scores}.pt", dry_run=dry_run)
    return OutputFolder(
        folder=output_folder,
        config=configf,
        params=paramsf,
        results=resultsf,
        basis=basisf,
        optim=optimf,
        scores=scoresf,
    )


def get_tracking_filename(run_id: str) -> Path:
    """Get filename for tracking file.

    Args:
        run_id (str): Run ID.

    Returns:
        Path: Tracking filename.
    """
    tracking_folder = Path.cwd() / "tracking"
    if not tracking_folder.is_dir():
        tracking_folder.mkdir(parents=True)
        gitignore = tracking_folder.joinpath(".gitignore")
        with gitignore.open("w") as file:
            file.write("*")
    return tracking_folder / f"track_{run_id}.json"


def track_output(output: OutputFolder, run_id: str | None) -> Path | None:
    """Output tracking file.

    Args:
        output (OutputFolder): Output folder.
        run_id (str | None): Run ID.

    Returns:
        Path | None: Path of tracking file.
    """
    if run_id is None:
        return None
    output_folder = output.folder.absolute()
    track_file = get_tracking_filename(run_id)
    msg = f"Output folder tracked through {track_file}"
    if output.dry_run:
        msg = "[dry-run] " + msg
    logger.info(msg)
    if output.dry_run:
        return None
    with track_file.open("w") as f:
        json.dump(
            {"folder": output_folder}, f, default=default_serializer, indent=4
        )
    return track_file


def retrieve_output_folder(
    output_folder: Path | None, run_id: str | None = None
) -> OutputFolder:
    """Retrieve output folder.

    Args:
        output_folder (Path | None): Output folder.
        run_id (str | None, optional): Run ID. Defaults to None.

    Raises:
        ValueError: IF both output and run_id are None.

    Returns:
        OutputFolder: Output folder.
    """
    if output_folder is None and run_id is None:
        msg = "Specify either an output folder or a run id."
        raise ValueError(msg)
    if output_folder is None:
        tracking_file = get_tracking_filename(run_id)
        with tracking_file.open("r") as f:
            output_folder = json.load(f)["folder"]
    output = build_output_folder(output_folder)
    msg = f"Loading outputs from {output_folder}."
    logger.info(box(msg))
    return output
