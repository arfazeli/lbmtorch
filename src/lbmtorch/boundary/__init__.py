"""Boundary conditions.

Periodicity is the default everywhere: streaming uses ``torch.roll``. Other
boundaries are applied after streaming and may overwrite populations.
"""

from lbmtorch.boundary.base import BoundaryCondition
from lbmtorch.boundary.bounce_back import BounceBack
from lbmtorch.boundary.interpolated import InterpolatedBounceBack
from lbmtorch.boundary.zou_he import ZouHePressureOutlet, ZouHeVelocityInlet

__all__ = [
    "BounceBack",
    "BoundaryCondition",
    "InterpolatedBounceBack",
    "ZouHePressureOutlet",
    "ZouHeVelocityInlet",
]
