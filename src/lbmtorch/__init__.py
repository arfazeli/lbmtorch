"""lbmtorch: a differentiable lattice Boltzmann solver in PyTorch.

A simulation is assembled from a :class:`~lbmtorch.lattice.Lattice`, a collision
operator, and a list of boundary conditions, then driven by
:class:`~lbmtorch.simulation.Simulation`.
"""

from lbmtorch.boundary import (
    BounceBack,
    InterpolatedBounceBack,
    ZouHePressureOutlet,
    ZouHeVelocityInlet,
)
from lbmtorch.collision import BGK, MRT, TRT, Smagorinsky
from lbmtorch.lattice import D2Q9, Lattice
from lbmtorch.macroscopic import density, equilibrium, velocity
from lbmtorch.simulation import Simulation, stream
from lbmtorch.units import tau_from_viscosity, viscosity_from_tau

__all__ = [
    "BGK",
    "BounceBack",
    "D2Q9",
    "InterpolatedBounceBack",
    "Lattice",
    "MRT",
    "Simulation",
    "Smagorinsky",
    "TRT",
    "ZouHePressureOutlet",
    "ZouHeVelocityInlet",
    "density",
    "equilibrium",
    "stream",
    "tau_from_viscosity",
    "velocity",
    "viscosity_from_tau",
]
