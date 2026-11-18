"""Multiple-relaxation-time collision for D2Q9 (Lallemand and Luo, 2000)."""

from __future__ import annotations

import torch

from lbmtorch.collision.bgk import guo_forcing
from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import equilibrium

# Rows: rho, e, eps, j_x, q_x, j_y, q_y, p_xx, p_xy  (Lallemand-Luo ordering).
# Columns follow the lbmtorch D2Q9 velocity ordering.
_M_D2Q9 = torch.tensor(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ],
    dtype=torch.float64,
)
# The rows of M are orthogonal, so M^{-1} = M^T diag(1 / |row|^2).
_MINV_D2Q9 = _M_D2Q9.T / (_M_D2Q9 * _M_D2Q9).sum(1)


class MRT:
    r"""Multiple-relaxation-time collision operator for D2Q9.

    Collision happens in moment space, :math:`\mathbf{m} = M \mathbf{f}`:

    .. math::

        \mathbf{f}^\star = \mathbf{f} - M^{-1} S \left(\mathbf{m} - \mathbf{m}^{eq}\right)
        + M^{-1}\left(I - \tfrac{1}{2}S\right) M\, \mathbf{S}^{\rm Guo},

    with :math:`S = \operatorname{diag}(s_0, \dots, s_8)`. The equilibrium moments are
    :math:`M \mathbf{f}^{eq}` with the standard second-order equilibrium, which matches
    the Lallemand-Luo choice up to :math:`O(u^3)`. Conserved moments (density and
    momentum) have rates that do not matter; the shear rate :math:`s_7 = s_8 = 1/\tau`
    sets the viscosity :math:`\nu = c_s^2(\tau - 1/2)`.

    Parameters
    ----------
    tau
        Shear relaxation time. May require grad.
    s_bulk, s_eps, s_flux
        Rates for the ``e``, ``eps`` and ``q`` moments; ``s_bulk`` sets the bulk viscosity.
        Defaults: Lallemand and Luo (2000), with ``s_flux`` lowered to 1.2 for stability.
        Any may be a tensor with grad.
    """

    def __init__(
        self,
        tau: float | torch.Tensor,
        s_bulk: float | torch.Tensor = 1.64,
        s_eps: float | torch.Tensor = 1.54,
        s_flux: float | torch.Tensor = 1.2,
    ):
        self.tau = tau
        self.s_bulk = s_bulk
        self.s_eps = s_eps
        self.s_flux = s_flux

    def rates(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Relaxation-rate vector ``(9,)`` in Lallemand-Luo moment order."""
        one = torch.ones((), dtype=dtype, device=device)
        s_nu = one / self.tau
        s = [
            one,
            one * self.s_bulk,
            one * self.s_eps,
            one,
            one * self.s_flux,
            one,
            one * self.s_flux,
            s_nu,
            s_nu,
        ]
        return torch.stack(s).to(dtype)  # a float64 tau must not promote a float32 run

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return post-collision populations."""
        if lattice.Q != 9 or lattice.D != 2:
            raise ValueError("MRT is implemented for the D2Q9 lattice only")
        M = _M_D2Q9.to(f.device, f.dtype)
        Minv = _MINV_D2Q9.to(f.device, f.dtype)
        s = self.rates(f.dtype, f.device)
        C = Minv @ (s[:, None] * M)
        f_post = f - torch.einsum("pq,q...->p...", C, f - equilibrium(rho, u, lattice))
        if force is not None:
            G = Minv @ ((1.0 - 0.5 * s)[:, None] * M)
            f_post = f_post + torch.einsum("pq,q...->p...", G, guo_forcing(u, force, lattice))
        return f_post
