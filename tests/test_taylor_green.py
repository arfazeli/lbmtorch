"""Taylor-Green vortex: viscous decay rate and second-order spatial convergence."""

import math

import torch

from lbmtorch import BGK, D2Q9, Simulation, viscosity_from_tau
from lbmtorch.benchmarks import taylor_green


def run(n, u_amp, tau, steps):
    nu = viscosity_from_tau(tau)
    u0, rho0 = taylor_green(n, u_amp, nu, 0.0)
    sim = Simulation(D2Q9, BGK(tau), (n, n), rho0=rho0, u0=u0, dtype=torch.float64)
    with torch.no_grad():
        sim.run(steps)
    u_ex, _ = taylor_green(n, u_amp, nu, steps)
    return sim.velocity, u_ex


def test_decay_rate():
    n, tau = 32, 0.8
    nu = viscosity_from_tau(tau)
    k = 2 * math.pi / n
    u0, rho0 = taylor_green(n, 0.01, nu, 0.0)
    sim = Simulation(D2Q9, BGK(tau), (n, n), rho0=rho0, u0=u0, dtype=torch.float64)
    ke = []
    with torch.no_grad():
        for _ in range(6):
            sim.run(20)
            ke.append((sim.velocity**2).sum().item())
    measured = [-math.log(ke[i + 1] / ke[i]) / 20 for i in range(len(ke) - 1)]
    expected = 4 * nu * k * k  # kinetic energy decays as exp(-4 nu k^2 t)
    for m in measured:
        assert abs(m - expected) / expected < 0.02, (m, expected)


def test_second_order_convergence():
    # Diffusive scaling: tau fixed, u_amp ~ 1/n, steps ~ n^2, so the physical end time is fixed.
    tau = 0.8
    errors = []
    for n in (16, 32, 64):
        u, u_ex = run(n, u_amp=0.02 * 16 / n, tau=tau, steps=16 * (n // 16) ** 2)
        errors.append(((u - u_ex) ** 2).sum().sqrt() / (u_ex**2).sum().sqrt())
    order = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(o > 1.8 for o in order), (errors, order)
