"""Spatial tools.

Warning: Since the first coordinate of the Tensor represents
the x coordinates, the actual Tensor is a 90° clockwise rotation
of the intuitive X,Y Grid.

Intuitive Representation for x and y values:

y                            y
^                            ^

:     :     :                :     :     :
x1----x2----x3..             y3----y3----y3..
|     |     |                |     |     |
|     |     |                |     |     |
x1----x2----x3..             y2----y2----y2..
|     |     |                |     |     |
|     |     |                |     |     |
x1----x2----x3..  >x         y1----y1----y1..  >x

Actual Implementation for x and y values:

x1----x1----x1..  >y         y1----y2----y3..  >y
|     |     |                |     |     |
|     |     |                |     |     |
x2----x2----x2..             y1----y2----y3..
|     |     |                |     |     |
|     |     |                |     |     |
x3----x3----x3..             y1----y2----y3..
:     :     :                :     :     :

v                            v
x                            x

The Space Discretization uses a staggered grid for:
- ω : Vorticity
- h : Layer Thickness Anomaly
- u : Zonal Velocity
- v : Meridional Velocity

The Intuitive Representation of this grid is the following:

y
^

:       :       :       :
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..   > x

While its actual implementation is:

ω---u---ω---u---ω---u---ω..   > y
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
:       :       :       :


v
x
"""

from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)

__all__ = [
    "SpaceDiscretization2D",
    "SpaceDiscretization3D",
]
