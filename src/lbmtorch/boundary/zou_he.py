"""Zou--He velocity inlet and pressure outlet for the D2Q9 x boundaries."""

from __future__ import annotations

import torch

from lbmtorch.lattice import Lattice


def _edge_values(f: torch.Tensor, x: int) -> list[torch.Tensor]:
    """Return the nine population rows at an x boundary."""
    return list(f[:, x, :].unbind(0))


def _replace_edge(f: torch.Tensor, x: int, replacements: dict[int, torch.Tensor]) -> torch.Tensor:
    """Replace selected boundary populations without modifying ``f`` in place."""
    out = list(f.unbind(0))
    for q, row in replacements.items():
        out[q] = (
            torch.cat((row.unsqueeze(0), out[q][1:]), dim=0)
            if x == 0
            else torch.cat((out[q][:-1], row.unsqueeze(0)), dim=0)
        )
    return torch.stack(out)


def _check_lattice(lattice: Lattice) -> None:
    if lattice.Q != 9 or lattice.D != 2:
        raise ValueError("Zou--He boundaries require the D2Q9 lattice")


class ZouHeVelocityInlet:
    r"""Prescribe velocity at the left (``x = 0``) edge of a D2Q9 domain.

    The unknown east-going populations are reconstructed by bounce-back of their
    non-equilibrium parts. The local density follows from the known populations:

    .. math::

        \rho = \frac{f_0+f_2+f_4+2(f_3+f_6+f_7)}{1-u_x}.

    Parameters
    ----------
    velocity
        Prescribed ``(u_x, u_y)``, shape ``(2,)`` or ``(2, ny)``. May require grad.

    Notes
    -----
    Intended for unforced flow: the boundary protocol passes no body force.
    """

    def __init__(self, velocity: torch.Tensor):
        self.prescribed_velocity = velocity

    def __call__(
        self, f_post: torch.Tensor, f_stream: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Reconstruct populations entering from the left edge after streaming."""
        _check_lattice(lattice)
        u = torch.as_tensor(self.prescribed_velocity, device=f_stream.device, dtype=f_stream.dtype)
        ux, uy = u.unbind(0)
        f = _edge_values(f_stream, 0)
        rho = (f[0] + f[2] + f[4] + 2 * (f[3] + f[6] + f[7])) / (1 - ux)
        return _replace_edge(
            f_stream,
            0,
            {
                1: f[3] + (2 / 3) * rho * ux,
                5: f[7] + 0.5 * (f[4] - f[2]) + rho * (ux / 6 + uy / 2),
                8: f[6] + 0.5 * (f[2] - f[4]) + rho * (ux / 6 - uy / 2),
            },
        )


class ZouHePressureOutlet:
    r"""Prescribe pressure at the right (``x = nx - 1``) edge of a D2Q9 domain.

    Pressure in lattice units is :math:`p=c_s^2\rho`. The normal velocity is
    recovered from the known populations, then the west-going populations are
    reconstructed by bounce-back of their non-equilibrium parts.

    Parameters
    ----------
    rho
        Prescribed density, scalar or shape ``(ny,)``. May require gradients.
    uy
        Prescribed transverse velocity, scalar or shape ``(ny,)``; defaults to zero.

    Notes
    -----
    Intended for unforced flow: the boundary protocol passes no body force.
    """

    def __init__(self, rho: float | torch.Tensor, uy: float | torch.Tensor = 0.0):
        self.rho = rho
        self.uy = uy

    def __call__(
        self, f_post: torch.Tensor, f_stream: torch.Tensor, lattice: Lattice
    ) -> torch.Tensor:
        """Reconstruct populations entering from the right edge after streaming."""
        _check_lattice(lattice)
        rho = torch.as_tensor(self.rho, device=f_stream.device, dtype=f_stream.dtype)
        uy = torch.as_tensor(self.uy, device=f_stream.device, dtype=f_stream.dtype)
        f = _edge_values(f_stream, -1)
        ux = (f[0] + f[2] + f[4] + 2 * (f[1] + f[5] + f[8])) / rho - 1
        return _replace_edge(
            f_stream,
            -1,
            {
                3: f[1] - (2 / 3) * rho * ux,
                6: f[8] + 0.5 * (f[4] - f[2]) + rho * (-ux / 6 + uy / 2),
                7: f[5] + 0.5 * (f[2] - f[4]) + rho * (-ux / 6 - uy / 2),
            },
        )
