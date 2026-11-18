"""Two-relaxation-time collision (Ginzburg, Verhaeghe and d'Humières, 2008)."""

from __future__ import annotations

import torch

from lbmtorch.collision.bgk import guo_forcing
from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import equilibrium


class TRT:
    r"""Two-relaxation-time collision.

    Populations are split into symmetric and antisymmetric parts,
    :math:`f_q^\pm = (f_q \pm f_{\bar q})/2`, relaxed at rates :math:`\omega^+ = 1/\tau^+`
    and :math:`\omega^- = 1/\tau^-`:

    .. math::

        f_q^\star = f_q - \omega^+ (f_q^+ - f_q^{eq,+}) - \omega^- (f_q^- - f_q^{eq,-})
        + \left(1 - \tfrac{\omega^+}{2}\right) S_q^+ + \left(1 - \tfrac{\omega^-}{2}\right) S_q^-.

    :math:`\tau^+` sets the viscosity, :math:`\nu = c_s^2(\tau^+ - 1/2)`. :math:`\tau^-` is
    fixed through the magic parameter
    :math:`\Lambda = (\tau^+ - 1/2)(\tau^- - 1/2)`. With :math:`\Lambda = 3/16` halfway
    bounce-back places the wall exactly midway for *any* viscosity, which removes the
    viscosity-dependent slip of BGK. :math:`\Lambda = 1/4` gives the best stability.
    Setting :math:`\tau^- = \tau^+` recovers BGK.

    Parameters
    ----------
    tau
        Symmetric relaxation time :math:`\tau^+ > 1/2`. May require grad.
    magic
        :math:`\Lambda`. Default 3/16.
    """

    def __init__(self, tau: float | torch.Tensor, magic: float = 3.0 / 16.0):
        self.tau = tau
        self.magic = magic

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return post-collision populations."""
        tau_p = self.tau
        tau_m = 0.5 + self.magic / (tau_p - 0.5)
        om_p, om_m = 1.0 / tau_p, 1.0 / tau_m
        opp = lattice.opp
        neq = f - equilibrium(rho, u, lattice)
        neq_p = 0.5 * (neq + neq[opp])
        neq_m = 0.5 * (neq - neq[opp])
        f_post = f - om_p * neq_p - om_m * neq_m
        if force is not None:
            s = guo_forcing(u, force, lattice)
            s_p, s_m = 0.5 * (s + s[opp]), 0.5 * (s - s[opp])
            f_post = f_post + (1.0 - 0.5 * om_p) * s_p + (1.0 - 0.5 * om_m) * s_m
        return f_post
