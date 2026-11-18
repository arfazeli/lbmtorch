"""Halfway bounce-back on an arbitrary solid mask, with optional wall motion."""

from __future__ import annotations

import torch

from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import expand_force


class BounceBack:
    r"""Halfway bounce-back no-slip walls, optionally moving.

    Any post-collision population at a fluid node that would stream into a solid node
    is instead reflected back to the same fluid node with its direction reversed:

    .. math::

        f_{\bar q}(\mathbf{x}, t+1) = f_q^\star(\mathbf{x}, t)
        - 2 w_q \rho(\mathbf{x}) \frac{\mathbf{c}_q \cdot \mathbf{u}_w}{c_s^2},

    where :math:`\mathbf{u}_w` is the wall velocity at the solid node
    :math:`\mathbf{x} + \mathbf{c}_q` (Ladd, 1994). The effective wall lies midway
    between the last fluid node and the first solid node.

    Solid nodes keep whatever populations stream into them. They are never read by
    fluid nodes, so their values are irrelevant, but they stay finite, which keeps
    autograd free of NaNs.

    Parameters
    ----------
    solid
        Boolean mask of shape ``(*spatial,)``; ``True`` marks solid nodes.
    velocity
        Wall velocity, shape ``(D,)`` for a uniform value or ``(D, *spatial)`` for a
        per-node field (only values at solid nodes are used). ``None`` means stationary.
    """

    def __init__(self, solid: torch.Tensor, velocity: torch.Tensor | None = None):
        if solid.dtype != torch.bool:
            raise TypeError("solid mask must be a boolean tensor")
        self.solid = solid
        self.velocity = velocity

    def _links(self, f_post: torch.Tensor, lattice: Lattice):
        """Yield ``(q, hit, reflected)``; ``hit[x]`` is True when ``x + c_q`` is solid."""
        solid = self.solid.to(f_post.device)
        dims = tuple(range(solid.ndim))
        if self.velocity is not None:
            u_w = expand_force(self.velocity.to(f_post.device, f_post.dtype), solid.ndim)
            u_w = u_w.expand(lattice.D, *solid.shape)
            rho = f_post.sum(0)  # collision conserves mass, so this is the node density
            c = lattice.c.to(f_post.dtype)
        for q in range(lattice.Q):
            shift = tuple(-s for s in lattice.shifts[q])
            hit = torch.roll(solid, shifts=shift, dims=dims)
            reflected = f_post[q]
            if self.velocity is not None:
                # wall velocity seen from the fluid node is the value at x + c_q
                u_wall = torch.roll(u_w, shifts=shift, dims=tuple(d + 1 for d in dims))
                cu = torch.einsum("d,d...->...", c[q], u_wall)
                reflected = reflected - 2.0 * lattice.w[q] * rho * cu / lattice.cs2
            yield q, hit, reflected

    def __call__(
        self, f_post: torch.Tensor, f_stream: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Apply reflection. See class docstring."""
        out = list(f_stream.unbind(0))
        for q, hit, reflected in self._links(f_post, lattice):
            qo = lattice.opp_idx[q]
            out[qo] = torch.where(hit, reflected, out[qo])
        return torch.stack(out)

    def force(self, f_post: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        r"""Total hydrodynamic force on the solid by momentum exchange (Ladd, 1994).

        .. math::

            \mathbf{F} = \sum_{\mathbf{x}\in\text{fluid}}
            \sum_{q:\,\mathbf{x}+\mathbf{c}_q\in\text{solid}}
            \mathbf{c}_q \left[ f_q^\star(\mathbf{x}) + f_{\bar q}(\mathbf{x}, t+1) \right]

        Pass the same post-collision populations that were streamed this step. Returns a
        ``(D,)`` tensor in lattice units, differentiable w.r.t. everything upstream.
        """
        fluid = ~self.solid.to(f_post.device)
        c = lattice.c.to(f_post.dtype)
        total = torch.zeros(lattice.D, dtype=f_post.dtype, device=f_post.device)
        for q, hit, reflected in self._links(f_post, lattice):
            hit = hit & fluid  # solid -> solid links carry junk populations
            exchanged = torch.where(hit, f_post[q] + reflected, torch.zeros_like(reflected)).sum()
            total = total + c[q] * exchanged
        return total
