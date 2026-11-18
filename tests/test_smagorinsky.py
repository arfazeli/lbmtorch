"""Smagorinsky LES: reduces to BGK for cs = 0 and in uniform flow; raises tau under shear."""

import torch

from lbmtorch import BGK, D2Q9, Simulation, Smagorinsky, equilibrium
from lbmtorch.macroscopic import density, velocity


def _shear_state(n=8):
    torch.manual_seed(0)
    y = torch.arange(n, dtype=torch.float64)
    u = torch.zeros(2, n, n, dtype=torch.float64)
    u[0] = 0.05 * torch.sin(2 * torch.pi * y / n)[None, :]
    rho = torch.ones(n, n, dtype=torch.float64)
    f = equilibrium(rho, u, D2Q9)
    # one BGK step generates a physical non-equilibrium part
    sim = Simulation(D2Q9, BGK(0.7), (n, n), u0=u, dtype=torch.float64)
    sim.run(1)
    f = sim.f
    return f, density(f), velocity(f, density(f), D2Q9)


def test_cs_zero_is_bgk():
    f, rho, u = _shear_state()
    assert torch.allclose(Smagorinsky(0.7, cs=0.0)(f, rho, u, D2Q9), BGK(0.7)(f, rho, u, D2Q9))


def test_uniform_flow_is_bgk():
    rho = torch.ones(5, 5, dtype=torch.float64)
    u = torch.full((2, 5, 5), 0.03, dtype=torch.float64)
    f = equilibrium(rho, u, D2Q9)
    assert torch.allclose(Smagorinsky(0.7, cs=0.2)(f, rho, u, D2Q9), BGK(0.7)(f, rho, u, D2Q9))


def test_shear_raises_tau_and_is_differentiable():
    f, rho, u = _shear_state()
    cs = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    model = Smagorinsky(0.7, cs=cs)
    tau_eff = model.effective_tau(f, equilibrium(rho, u, D2Q9), rho, D2Q9)
    assert (tau_eff >= 0.7).all() and tau_eff.max() > 0.7
    (g,) = torch.autograd.grad(tau_eff.sum(), cs)
    assert torch.isfinite(g) and g > 0
