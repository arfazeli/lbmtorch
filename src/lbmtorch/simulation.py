"""Time-stepping driver."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from lbmtorch.boundary.base import BoundaryCondition
from lbmtorch.collision.base import CollisionOperator
from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import density, equilibrium, velocity


def stream(f: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    """Periodic streaming: ``f_q(x + c_q) <- f_q(x)`` for every direction."""
    dims = tuple(range(f.ndim - 1))
    return torch.stack(
        [torch.roll(f[q], shifts=lattice.shifts[q], dims=dims) for q in range(lattice.Q)]
    )


class Simulation:
    """Collide-stream-boundary loop on a structured lattice.

    Parameters
    ----------
    lattice
        Velocity set, e.g. :data:`lbmtorch.D2Q9`.
    collision
        Collision operator, e.g. :class:`lbmtorch.BGK`.
    shape
        Spatial grid shape ``(nx, ny)``.
    boundaries
        Boundary conditions applied in order after streaming.
    force
        Body force density, shape ``(D,)`` or ``(D, *shape)``. May require grad.
    rho0, u0
        Initial density (scalar or field) and velocity field ``(D, *shape)``.
        The initial populations are the corresponding equilibrium.
    device, dtype
        Where and in what precision to run. Defaults to CPU, ``float32``.
    """

    def __init__(
        self,
        lattice: Lattice,
        collision: CollisionOperator,
        shape: Sequence[int],
        boundaries: Sequence[BoundaryCondition] = (),
        force: torch.Tensor | None = None,
        rho0: float | torch.Tensor = 1.0,
        u0: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        self.lattice = lattice.to(self.device, dtype)
        self.collision = collision
        self.boundaries = list(boundaries)
        self.force = None if force is None else force.to(self.device, dtype)
        self.shape = tuple(shape)
        self.step_count = 0

        rho = torch.as_tensor(rho0, device=self.device, dtype=dtype).expand(self.shape)
        u = (
            torch.zeros(self.lattice.D, *self.shape, device=self.device, dtype=dtype)
            if u0 is None
            else u0.to(self.device, dtype)
        )
        self.f = equilibrium(rho, u, self.lattice)

    def advance(self, f: torch.Tensor, n_steps: int = 1) -> torch.Tensor:
        """Return the populations ``n_steps`` cycles after ``f``, without mutating ``self``."""
        for _ in range(n_steps):
            rho = density(f)
            u = velocity(f, rho, self.lattice, self.force)
            f_post = self.collision(f, rho, u, self.lattice, self.force)
            f_new = stream(f_post, self.lattice)
            for bc in self.boundaries:
                f_new = bc(f_post, f_new, self.lattice)
            f = f_new
        return f

    def step(self) -> None:
        """Advance one lattice time step."""
        self.f = self.advance(self.f)
        self.step_count += 1

    def run(self, n_steps: int, checkpoint_every: int | None = None) -> None:
        """Advance ``n_steps`` time steps.

        Parameters
        ----------
        n_steps
            Number of lattice time steps.
        checkpoint_every
            If set and gradients are enabled, wrap every ``checkpoint_every`` steps in
            :func:`torch.utils.checkpoint.checkpoint`. Autograd then stores only the
            populations at chunk boundaries and recomputes the chunk interior during the
            backward pass. Memory drops from :math:`O(n_{steps})` to
            :math:`O(n_{steps}/k + k)` at the cost of one extra forward pass.
        """
        if checkpoint_every is None or not torch.is_grad_enabled():
            for _ in range(n_steps):
                self.step()
            return
        done = 0
        while done < n_steps:
            n = min(checkpoint_every, n_steps - done)
            self.f = checkpoint(self.advance, self.f, n, use_reentrant=False)
            self.step_count += n
            done += n

    @property
    def density(self) -> torch.Tensor:
        """Current density field, ``(*shape,)``."""
        return density(self.f)

    @property
    def velocity(self) -> torch.Tensor:
        """Current velocity field, ``(D, *shape)``, including the half-force shift."""
        return velocity(self.f, self.density, self.lattice, self.force)
