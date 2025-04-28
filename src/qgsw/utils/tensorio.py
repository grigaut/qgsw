"""Input / output methods for tensors."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw.exceptions import InvalidSavingFileError
from qgsw.specs import defaults


def raise_if_invalid_savefile(file: Path) -> None:
    """Raise an error if the saving file is invalid.

    Args:
        file (Path): Output file.

    Raises:
        InvalidSavingFileError: if the saving file extension is not .pt.
    """
    if file.suffix != ".pt":
        msg = "Variables are expected to be saved in an .pt file."
        raise InvalidSavingFileError(msg)


def preprocess_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess the tensor to be saved.

    Args:
        tensor (torch.Tensor): Tensor to preprocess.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    return tensor


def save(
    tensors: dict[str, torch.Tensor],
    file: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Save tensors to a gievn file.

    Args:
        tensors (dict[str, torch.Tensor]): Tensors to save.
        file (str | Path): Output file path.
        dtype (torch.dtype | None): Data type to use to save the tensor.
            Defaults to None.
        device (torch.device | None): Device to use to save the tensor.
            Defaults to None.
    """
    f = Path(file)
    raise_if_invalid_savefile(f)
    preprocessed = {
        k: v.to(**defaults.get_save_specs(dtype=dtype, device=device))
        for k, v in tensors.items()
    }
    torch.save(preprocessed, f)


def load(
    file: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Load tensors from a given file.

    Args:
        file (str | Path): Output file path.
        dtype (torch.dtype | None): Data type of the tensors to load.
            Defaults to None.
        device (torch.device | None): Device of the tensors to load.
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Loaded tensors.
    """
    f = Path(file)
    raise_if_invalid_savefile(f)
    tensors: dict[str, torch.Tensor] = torch.load(f)
    return {
        k: v.to(**defaults.get(dtype=dtype, device=device))
        for k, v in tensors.items()
    }
