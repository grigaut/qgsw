"""Sorting related methods."""

from pathlib import Path


def sort_files(files: list[Path], prefix: str, suffix: str) -> list[Path]:
    """Sort files matching prefix{nb}suffix (ignore '{' and '}').

    Args:
        files (list[Path]): Files list.
        prefix (str): Name Prefix.
        suffix (str): Name Suffix.

    Returns:
        list[Path]: Sorted files.
    """
    len_pre = len(prefix)
    len_suf = len(suffix)
    map_nb_files = {int(e.name[len_pre:-len_suf]): e for e in files}
    nbs = list(map_nb_files.keys())
    nbs.sort()
    return [map_nb_files[nb] for nb in nbs]
