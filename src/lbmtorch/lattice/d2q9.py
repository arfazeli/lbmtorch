"""The D2Q9 lattice.

Velocity ordering (rest, axis-aligned, diagonals)::

    6 2 5
    3 0 1
    7 4 8
"""

import torch

from lbmtorch.lattice.base import Lattice, opposite_indices

_C = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.long,
)
_W = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float64)

D2Q9 = Lattice(name="D2Q9", c=_C, w=_W, opp=opposite_indices(_C))
"""Two-dimensional, nine-velocity lattice with :math:`c_s^2 = 1/3`."""
