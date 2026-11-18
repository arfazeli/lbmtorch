"""Interpolated bounce-back for curved walls (Bouzidi, Firdaouss and Lallemand, 2001)."""

from __future__ import annotations

import torch

from lbmtorch.lattice import Lattice


class InterpolatedBounceBack:
    r"""Second-order no-slip walls at arbitrary positions along lattice links.

    The wall is described by a signed distance field :math:`\phi(\mathbf{x})`, negative
    inside the solid. For each fluid node :math:`\mathbf{x}_f` and direction :math:`q`
    whose neighbour :math:`\mathbf{x}_s = \mathbf{x}_f + \mathbf{c}_q` is solid, the
    fraction of the link inside the fluid is estimated by linear interpolation,
    :math:`\delta = \phi(\mathbf{x}_f)/(\phi(\mathbf{x}_f) - \phi(\mathbf{x}_s))`, and the
    reflected population is (Bouzidi et al. 2001, linear scheme)

    .. math::

        f_{\bar q}(\mathbf{x}_f, t+1) = \begin{cases}
        2\delta\, f_q^\star(\mathbf{x}_f) + (1 - 2\delta)\, f_q^\star(\mathbf{x}_f - \mathbf{c}_q),
        & \delta < \tfrac12 \\[4pt]
        \dfrac{1}{2\delta}\, f_q^\star(\mathbf{x}_f)
        + \dfrac{2\delta - 1}{2\delta}\, f_{\bar q}^\star(\mathbf{x}_f),
        & \delta \ge \tfrac12
        \end{cases}

    :math:`\delta = 1/2` everywhere reduces to halfway bounce-back. Because :math:`\phi`
    is a tensor, gradients flow to the wall position, which is what shape optimisation
    needs. The upstream node :math:`\mathbf{x}_f - \mathbf{c}_q` is assumed to be fluid;
    at one-node-wide gaps the scheme silently degrades.

    Parameters
    ----------
    phi
        Signed distance (or any level set), shape ``(*spatial,)``. Solid where
        ``phi < 0``. May require grad.
    """

    def __init__(self, phi: torch.Tensor):
        self.phi = phi

    @property
    def solid(self) -> torch.Tensor:
        """Boolean solid mask derived from the level set."""
        return self.phi < 0

    def _links(self, q: int, lattice: Lattice, f_post: torch.Tensor, phi: torch.Tensor):
        """Return (hit mask, reflected population) for direction ``q``."""
        solid = phi < 0
        dims = tuple(range(phi.ndim))
        shift = tuple(-s for s in lattice.shifts[q])  # value at x + c_q, seen from x
        back = lattice.shifts[q]  # value at x - c_q, seen from x
        hit = torch.roll(solid, shifts=shift, dims=dims) & ~solid
        phi_s = torch.roll(phi, shifts=shift, dims=dims)
        # clamp keeps delta in (0, 1] and the division finite away from the wall
        delta = (phi / (phi - phi_s).clamp_min(1e-12)).clamp(1e-6, 1.0)
        fq, fqo = f_post[q], f_post[lattice.opp_idx[q]]
        fq_up = torch.roll(fq, shifts=back, dims=dims)
        near = 2 * delta * fq + (1 - 2 * delta) * fq_up
        far = fq / (2 * delta) + (2 * delta - 1) / (2 * delta) * fqo
        return hit, torch.where(delta < 0.5, near, far)

    def __call__(
        self, f_post: torch.Tensor, f_stream: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Apply interpolated reflection. See class docstring."""
        phi = self.phi.to(f_post.device, f_post.dtype)
        out = list(f_stream.unbind(0))
        for q in range(lattice.Q):
            hit, reflected = self._links(q, lattice, f_post, phi)
            qo = lattice.opp_idx[q]
            out[qo] = torch.where(hit, reflected, out[qo])
        return torch.stack(out)

    def force(self, f_post: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        """Momentum-exchange force on the solid, ``(D,)``. See :meth:`BounceBack.force`."""
        phi = self.phi.to(f_post.device, f_post.dtype)
        c = lattice.c.to(f_post.dtype)
        total = torch.zeros(lattice.D, dtype=f_post.dtype, device=f_post.device)
        for q in range(lattice.Q):
            hit, reflected = self._links(q, lattice, f_post, phi)
            exchanged = torch.where(hit, f_post[q] + reflected, torch.zeros_like(reflected)).sum()
            total = total + c[q] * exchanged
        return total
