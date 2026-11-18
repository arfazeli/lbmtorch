"""Interface every boundary condition implements."""

from __future__ import annotations

from typing import Protocol

import torch

from lbmtorch.lattice import Lattice


class BoundaryCondition(Protocol):
    """Callable applied after streaming.

    Receives both the post-collision populations ``f_post`` (before streaming) and the
    streamed populations ``f_stream`` so that reflecting conditions can be expressed
    without extra copies. Returns the corrected populations.
    """

    def __call__(
        self, f_post: torch.Tensor, f_stream: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Return corrected populations with the same shape as ``f_stream``."""
        ...
