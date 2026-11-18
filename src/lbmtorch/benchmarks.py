"""Reference problem setups shared by the examples and the validation tests."""

from __future__ import annotations

import math

import torch

from lbmtorch import (
    D2Q9,
    MRT,
    BounceBack,
    InterpolatedBounceBack,
    Simulation,
    ZouHePressureOutlet,
    ZouHeVelocityInlet,
)
from lbmtorch.geometry import channel_walls
from lbmtorch.units import tau_from_viscosity


def taylor_green(
    n: int, u_amp: float, nu: float, t: float, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Decaying Taylor-Green vortex on an ``n`` x ``n`` periodic grid.

    .. math::

        \mathbf{u} = U e^{-2\nu k^2 t}(-\cos kx \sin ky,\ \sin kx \cos ky), \qquad
        \rho = 1 - \frac{U^2 e^{-4\nu k^2 t}}{4 c_s^2}(\cos 2kx + \cos 2ky),

    with :math:`k = 2\pi/n`. Returns ``(u, rho)`` with shapes ``(2, n, n)`` and ``(n, n)``.
    """
    k = 2 * math.pi / n
    x = torch.arange(n, dtype=dtype)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    decay = math.exp(-2 * nu * k * k * t)
    u = (
        u_amp
        * decay
        * torch.stack([-torch.cos(k * X) * torch.sin(k * Y), torch.sin(k * X) * torch.cos(k * Y)])
    )
    rho = 1 - (u_amp * decay) ** 2 / (4 * D2Q9.cs2) * (torch.cos(2 * k * X) + torch.cos(2 * k * Y))
    return u, rho


def schaefer_turek_cylinder(
    diameter: int = 20,
    u_mean: float = 0.05,
    re: float = 100.0,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    curved: bool = True,
) -> tuple[Simulation, BounceBack | InterpolatedBounceBack]:
    r"""Schäfer and Turek (1996) benchmark 2D-2: cylinder in a channel at Re = 100.

    Geometry in units of the cylinder diameter :math:`D`: channel length 22, height 4.1,
    cylinder centre at (2, 2). The slight vertical offset breaks symmetry and triggers
    vortex shedding. The inlet is parabolic with mean speed ``u_mean``; the outlet holds
    :math:`\rho = 1`. Reference values: Strouhal 0.295 to 0.305, drag coefficient
    3.22 to 3.24, maximum lift coefficient 0.99 to 1.01.

    With ``curved=True`` (default) the cylinder uses :class:`InterpolatedBounceBack` on
    the exact signed distance, otherwise a staircase halfway bounce-back.

    Returns the simulation and the cylinder's bounce-back object, whose
    :meth:`~lbmtorch.BounceBack.force` gives the hydrodynamic force on the cylinder alone.
    The channel walls are a separate bounce-back object so their friction is excluded.
    """
    D = diameter
    nx, ny = round(22 * D), round(4.1 * D) + 2  # +2 for the solid wall rows
    cx, cy = 2 * D, 2 * D + 0.5  # +0.5: wall sits half a cell above the first fluid row
    x = torch.arange(nx, dtype=torch.float64)[:, None]
    y = torch.arange(ny, dtype=torch.float64)[None, :]
    phi = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) - D / 2  # signed distance, < 0 inside
    walls = channel_walls((nx, ny))

    h = ny - 2  # channel height in lattice units
    y_fluid = torch.arange(1, ny - 1, dtype=torch.float64) - 0.5  # 0 at the bottom wall
    profile = 4 * 1.5 * u_mean * y_fluid * (h - y_fluid) / h**2  # max = 1.5 * mean
    inlet = torch.zeros(2, ny, dtype=torch.float64)
    inlet[0, 1:-1] = profile

    tau = tau_from_viscosity(u_mean * D / re)
    phi, walls = phi.to(device, dtype), walls.to(device)  # avoid a host copy every step
    body = InterpolatedBounceBack(phi) if curved else BounceBack(phi < 0)
    sim = Simulation(
        D2Q9,
        MRT(tau),
        (nx, ny),
        boundaries=[
            ZouHeVelocityInlet(inlet.to(dtype)),
            ZouHePressureOutlet(1.0),
            BounceBack(walls),
            body,
        ],
        u0=inlet[:, None, :].expand(2, nx, ny).to(dtype),
        device=device,
        dtype=dtype,
    )
    return sim, body


def force_coefficients(force: torch.Tensor, u_mean: float, diameter: float) -> tuple[float, float]:
    r"""Drag and lift coefficients :math:`C = 2F/(\rho U^2 D)` with :math:`\rho = 1`.

    Returns Python floats, so the result is not differentiable.
    """
    scale = 2.0 / (u_mean**2 * diameter)
    return float(force[0]) * scale, float(force[1]) * scale


def strouhal_from_signal(signal: list[float], dt: int, u_mean: float, diameter: float) -> float:
    """Strouhal number from mean upward zero crossings of a periodic signal sampled every ``dt``."""
    s = torch.as_tensor(signal, dtype=torch.float64)
    s = s - s.mean()
    up = torch.nonzero((s[:-1] < 0) & (s[1:] >= 0)).flatten()
    if len(up) < 2:
        return math.nan
    period = float(up[-1] - up[0]) / (len(up) - 1) * dt
    return diameter / (period * u_mean)
