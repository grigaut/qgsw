"""Verbose.

Recommended ways to import this module is:
>>> from qgsw import verbose

Therefore, one can access verbose level with:
>>> from qgsw import verbose
>>> print(verbose.get_level())

And one can set verbose levl with:
>>> from qgsw import verbose
>>> verbose.set_level(2)
"""

from qgsw.verbose._core import (
    display,
    get_level,
    is_mute,
    set_level,
    set_prefix,
)

__all__ = [
    "display",
    "get_level",
    "is_mute",
    "set_level",
    "set_prefix",
]
