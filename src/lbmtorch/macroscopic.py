"""Moments of the distribution function and the equilibrium distribution."""

from __future__ import annotations

import torch

from lbmtorch.lattice import Lattice


def density(f: torch.Tensor) -> torch.Tensor:
    r"""Zeroth moment :math:`\rho = \sum_q f_q`, shape ``(*spatial,)``."""
    return f.sum(0)


def momentum(f: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    r"""First moment :math:`\rho u = \sum_q c_q f_q`, shape ``(D, *spatial)``."""
    return torch.einsum("qd,q...->d...", lattice.c.to(f.dtype), f)


def velocity(
    f: torch.Tensor,
    rho: torch.Tensor,
    lattice: Lattice,
    force: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Macroscopic velocity, including the half-force shift of the Guo forcing scheme.

    Parameters
    ----------
    f
        Populations, ``(Q, *spatial)``.
    rho
        Density, ``(*spatial,)``.
    lattice
        Velocity set.
    force
        Body force density, broadcastable to ``(D, *spatial)``. If given, the
        velocity is :math:`u = (\sum_q c_q f_q + F/2)/\rho`.
    """
    mom = momentum(f, lattice)
    if force is not None:
        mom = mom + 0.5 * expand_force(force, f.ndim - 1)
    return mom / rho


def equilibrium(rho: torch.Tensor, u: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    r"""Second-order Maxwell-Boltzmann equilibrium.

    .. math::

        f_q^{eq} = w_q \rho \left[ 1 + \frac{c_q\cdot u}{c_s^2}
        + \frac{(c_q\cdot u)^2}{2 c_s^4} - \frac{u\cdot u}{2 c_s^2} \right]
    """
    cs2 = lattice.cs2
    cu = torch.einsum("qd,d...->q...", lattice.c.to(u.dtype), u)
    usq = (u * u).sum(0)
    w = lattice.w_bcast(u.ndim - 1).to(u.dtype)
    return w * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs2 * cs2) - usq / (2.0 * cs2))


def expand_force(force: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a ``(D,)`` uniform force to ``(D, 1, ..., 1)``; pass fields through unchanged."""
    if force.ndim == 1:
        return force.view(-1, *([1] * ndim))
    return force
