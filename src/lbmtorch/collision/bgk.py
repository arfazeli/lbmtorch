"""Single-relaxation-time (BGK) collision with Guo forcing."""

from __future__ import annotations

import torch

from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import equilibrium, expand_force


class BGK:
    r"""Bhatnagar-Gross-Krook collision operator.

    .. math::

        f_q^\star = f_q - \frac{1}{\tau}\,(f_q - f_q^{eq})
        + \left(1 - \frac{1}{2\tau}\right) S_q

    where :math:`S_q` is the Guo et al. (2002) forcing term. The kinematic viscosity is
    :math:`\nu = c_s^2 (\tau - 1/2)`.

    Parameters
    ----------
    tau
        Relaxation time in lattice units. May be a tensor with ``requires_grad=True``
        to differentiate with respect to viscosity.
    """

    def __init__(self, tau: float | torch.Tensor):
        self.tau = tau

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return post-collision populations."""
        omega = 1.0 / self.tau
        f_post = f - omega * (f - equilibrium(rho, u, lattice))
        if force is not None:
            f_post = f_post + (1.0 - 0.5 * omega) * guo_forcing(u, force, lattice)
        return f_post


def guo_forcing(u: torch.Tensor, force: torch.Tensor, lattice: Lattice) -> torch.Tensor:
    r"""Guo forcing term.

    .. math::

        S_q = w_q \left[ \frac{c_q - u}{c_s^2}
        + \frac{(c_q\cdot u)\, c_q}{c_s^4} \right] \cdot F
    """
    cs2 = lattice.cs2
    ndim = u.ndim - 1
    c = lattice.c.to(u.dtype)
    F = expand_force(force, ndim)
    cu = torch.einsum("qd,d...->q...", c, u)
    cF = torch.einsum("qd,d...->q...", c, F)
    uF = (u * F).sum(0)
    w = lattice.w_bcast(ndim).to(u.dtype)
    return w * (cF / cs2 - uF / cs2 + cu * cF / (cs2 * cs2))
