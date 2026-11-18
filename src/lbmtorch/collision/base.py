"""Interface every collision operator implements."""

from __future__ import annotations

from typing import Protocol

import torch

from lbmtorch.lattice import Lattice


class CollisionOperator(Protocol):
    """Callable mapping pre-collision to post-collision populations.

    Implementations must be pure tensor functions so that autograd can trace them.
    """

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return post-collision populations with the same shape as ``f``."""
        ...
