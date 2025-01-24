"""Modified QG model with filtered top layer."""

from __future__ import annotations

from qgsw.fields.variables.uvh import UVHTAlpha
from qgsw.models.qg.core import QGCore


class QGCollinearFilteredSF(QGCore[UVHTAlpha]):
    """Modified QG Model implementing collinear pv behavior."""

    _type = "QGCollinearFilteredSF"

    def __init__(self) -> None:
        """InStantiate.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
