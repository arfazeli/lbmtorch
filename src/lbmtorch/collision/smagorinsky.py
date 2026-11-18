"""BGK with a Smagorinsky sub-grid eddy viscosity (Hou et al., 1996)."""

from __future__ import annotations

import torch

from lbmtorch.collision.bgk import guo_forcing
from lbmtorch.lattice import Lattice
from lbmtorch.macroscopic import equilibrium


class Smagorinsky:
    r"""Large-eddy BGK collision with a local, strain-dependent relaxation time.

    The non-equilibrium momentum flux
    :math:`\Pi^{neq}_{\alpha\beta} = \sum_q c_{q\alpha} c_{q\beta}(f_q - f_q^{eq})` is
    proportional to the resolved strain rate, so the Smagorinsky eddy viscosity
    :math:`\nu_t = (C_s \Delta)^2 |S|` can be folded into the relaxation time without
    finite differences:

    .. math::

        \tau_{\rm eff} = \tfrac{1}{2}\left[\tau_0
        + \sqrt{\tau_0^2 + 18\sqrt{2}\, C_s^2\, |\Pi^{neq}| / \rho}\,\right],
        \qquad |\Pi^{neq}| = \sqrt{\Pi^{neq}_{\alpha\beta}\Pi^{neq}_{\alpha\beta}},

    with :math:`\Delta = 1` and :math:`|S| = \sqrt{2 S_{\alpha\beta} S_{\alpha\beta}}`.
    Setting ``cs = 0`` recovers BGK exactly.

    Parameters
    ----------
    tau0
        Molecular relaxation time. May require grad.
    cs
        Smagorinsky constant, typically 0.1 to 0.2. May require grad, which makes it a
        one-parameter learnable closure.
    """

    def __init__(self, tau0: float | torch.Tensor, cs: float | torch.Tensor = 0.1):
        self.tau0 = tau0
        self.cs = cs

    def effective_tau(
        self, f: torch.Tensor, feq: torch.Tensor, rho: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Local relaxation time field, shape ``(*spatial,)``."""
        c = lattice.c.to(f.dtype)
        pi = torch.einsum("qa,qb,q...->ab...", c, c, f - feq)
        # eps keeps d|Pi|/dPi finite at Pi = 0 (uniform flow)
        pi_mag = torch.sqrt((pi * pi).sum((0, 1)) + 1e-30)
        tau0 = torch.as_tensor(self.tau0, dtype=f.dtype, device=f.device)
        cs = torch.as_tensor(self.cs, dtype=f.dtype, device=f.device)
        return 0.5 * (tau0 + torch.sqrt(tau0 * tau0 + 18.0 * 2.0**0.5 * cs * cs * pi_mag / rho))

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: torch.Tensor,
        lattice: Lattice,
        force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply collision with the local effective relaxation time."""
        feq = equilibrium(rho, u, lattice)
        omega = 1.0 / self.effective_tau(f, feq, rho, lattice)
        f_post = f - omega * (f - feq)
        if force is not None:
            f_post = f_post + (1.0 - 0.5 * omega) * guo_forcing(u, force, lattice)
        return f_post
