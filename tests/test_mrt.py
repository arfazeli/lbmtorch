"""MRT collision: reduces to BGK when all rates are equal, and validates on Poiseuille."""

import torch

from lbmtorch import BGK, D2Q9, MRT, Simulation, density, equilibrium, velocity
from test_poiseuille import poiseuille_setup


def test_mrt_equals_bgk_for_uniform_rates():
    torch.manual_seed(0)
    tau = 0.8
    rho = 1.0 + 0.05 * torch.rand(6, 5, dtype=torch.float64)
    u = 0.05 * (torch.rand(2, 6, 5, dtype=torch.float64) - 0.5)
    f = equilibrium(rho, u, D2Q9) * (1 + 0.01 * torch.randn(9, 6, 5, dtype=torch.float64))
    force = torch.tensor([1e-4, -2e-4], dtype=torch.float64)
    r, v = density(f), velocity(f, density(f), D2Q9, force)
    mrt = MRT(tau, s_bulk=1 / tau, s_eps=1 / tau, s_flux=1 / tau)
    assert torch.allclose(mrt(f, r, v, D2Q9, force), BGK(tau)(f, r, v, D2Q9, force), atol=1e-13)


def test_mrt_poiseuille():
    sim, u_exact = poiseuille_setup(ny_fluid=16, tau=0.6)
    sim.collision = MRT(0.6)
    with torch.no_grad():
        sim.run(6000)
    ux = sim.velocity[0, 0, 1:-1]
    # Halfway bounce-back slip error at tau = 0.6 is O(1e-2); MRT with these rates ~ 2%.
    assert (ux - u_exact).abs().max() / u_exact.max() < 0.03


def test_mrt_gradcheck_tau():
    def ke(tau):
        u0 = 1e-2 * torch.sin(torch.linspace(0, 6.28, 6, dtype=torch.float64))[None, :, None]
        u0 = torch.cat([u0.expand(1, 6, 4), torch.zeros(1, 6, 4, dtype=torch.float64)])
        sim = Simulation(D2Q9, MRT(tau), (6, 4), u0=u0, dtype=torch.float64)
        sim.run(4)
        return (sim.velocity**2).sum()

    tau = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ke, (tau,), eps=1e-6, atol=1e-8)
