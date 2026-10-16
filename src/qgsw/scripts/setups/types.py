"""Types."""

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction

Bases = dict[
    str, SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]
]
Coefs = dict[str, DecompositionCoefs]
