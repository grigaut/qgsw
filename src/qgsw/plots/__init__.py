"""Plotting related tools."""

import seaborn as sns

from qgsw.plots.plt_wrapper import (
    blittable_suptitle,
    clamp_ylims,
    close,
    imshow,
    imshow_,
    set_coltitles,
    set_rowtitles,
    set_ylims,
    show,
    subplots,
)

__all__ = [
    "blittable_suptitle",
    "clamp_ylims",
    "close",
    "imshow",
    "imshow_",
    "set_coltitles",
    "set_rowtitles",
    "set_ylims",
    "show",
    "subplots",
]
sns.set_theme("notebook", style="dark")
