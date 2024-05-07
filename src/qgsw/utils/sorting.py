"""Sorting related methods."""

from pathlib import Path


def extract_step_nb(file: Path, prefix: str, suffix: str) -> int:
    """Extract step number from file path.

    Args:
        file (Path): File
        prefix (str): Prefix
        suffix (str): Suffix

    Returns:
        int: Step
    """
    return int(file.name[len(prefix) : -len(suffix)])


def sort_files(
    files: list[Path],
    prefix: str,
    suffix: str,
) -> tuple[list[int], list[Path]]:
    """Sort files matching prefix{nb}suffix (ignore '{' and '}').

    Args:
        files (list[Path]): Files list.
        prefix (str): Name Prefix.
        suffix (str): Name Suffix.

    Returns:
        list[Path]: Sorted files.
    """
    map_step_file = {extract_step_nb(f, prefix, suffix): f for f in files}
    steps = list(map_step_file.keys())
    steps.sort()
    return steps, [map_step_file[step] for step in steps]
