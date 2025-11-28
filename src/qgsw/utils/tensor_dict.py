"""Utils for dictionaries with tensors."""

from __future__ import annotations

from typing import Any

import torch

from qgsw.specs import defaults


def change_specs(
    d: dict[str, Any],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Change specs of tensors in dictionnary.

    Args:
        d (dict[str, Any]): Dictionary to modify (inplace).
        dtype (torch.dtype | None, optional): Data type. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        dict[str, Any]: Modified dictionary.
    """
    specs = defaults.get(dtype=dtype, device=device)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(**specs)
        elif isinstance(v, dict):
            d[k] = change_specs(v, **specs)
    return d
