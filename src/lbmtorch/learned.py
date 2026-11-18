"""Learned collision: MRT plus a neural correction of the non-conserved moments."""

from __future__ import annotations

import torch
from torch import nn

from lbmtorch.collision.mrt import _M_D2Q9, MRT
from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import equilibrium

_NON_CONSERVED = [1, 2, 4, 6, 7, 8]  # e, eps, q_x, q_y, p_xx, p_xy in Lallemand-Luo order


class LearnedMRT(nn.Module):
    r"""MRT collision with a learned, conservative, local correction.

    .. math::

        \mathbf{f}^\star = \mathrm{MRT}(\mathbf{f}) + M^{-1}\,\delta\mathbf{m}(\mathbf{z}),
        \qquad \delta m_\rho = \delta m_{j_x} = \delta m_{j_y} = 0,

    where :math:`\delta\mathbf{m}` is produced by a small pointwise MLP from local
    features :math:`\mathbf{z} = (\rho - 1,\ u_x,\ u_y,\ \Pi^{neq}_{xx},\ \Pi^{neq}_{xy},
    \Pi^{neq}_{yy})`. Only non-conserved moments are touched, so mass and momentum are
    conserved exactly by construction, whatever the network does. The last layer is
    zero-initialised, so an untrained model is exactly MRT.

    Train it *solver-in-the-loop*: unroll :class:`~lbmtorch.Simulation` for some steps,
    compare with a reference, and backpropagate through the whole trajectory.

    Parameters
    ----------
    tau
        Shear relaxation time of the underlying MRT. May require grad.
    hidden
        Width of the two hidden layers.
    feature_scale, output_scale
        Fixed normalisation so the network sees O(1) inputs and emits O(1) outputs.
        The defaults suit Taylor-Green flow at :math:`U \sim 0.05`.
    """

    def __init__(
        self,
        tau: float | torch.Tensor,
        hidden: int = 32,
        feature_scale: tuple[float, ...] = (50.0, 10.0, 10.0, 1e3, 1e3, 1e3),
        output_scale: float = 1e-3,
    ):
        super().__init__()
        self.mrt = MRT(tau)
        self.net = nn.Sequential(
            nn.Linear(len(feature_scale), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(_NON_CONSERVED)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("feature_scale", torch.tensor(feature_scale, dtype=torch.float64))
        self.register_buffer("basis", torch.linalg.inv(_M_D2Q9)[:, _NON_CONSERVED])
        self.output_scale = output_scale

    def features(
        self, f: torch.Tensor, rho: torch.Tensor, u: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Local invariants, shape ``(6, *spatial)``."""
        c = lattice.c.to(f.dtype)
        pi = torch.einsum("qa,qb,q...->ab...", c, c, f - equilibrium(rho, u, lattice))
        z = torch.stack([rho - 1.0, u[0], u[1], pi[0, 0], pi[0, 1], pi[1, 1]])
        return z * self.feature_scale.to(f.dtype).view(-1, *([1] * (f.ndim - 1)))

    def forward(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply MRT, then add the learned conservative correction."""
        f_post = self.mrt(f, rho, u, lattice, force)
        z = self.features(f, rho, u, lattice).movedim(0, -1)  # (*spatial, 6)
        dm = self.net(z.to(self.net[0].weight.dtype)).to(f.dtype).movedim(-1, 0)  # (6, *spatial)
        basis = self.basis.to(f.device, f.dtype)
        corr = torch.einsum("pq,q...->p...", basis, dm) * self.output_scale
        return f_post + corr
