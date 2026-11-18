"""TRT: equals BGK when tau- = tau+; magic 3/16 gives an exact Poiseuille profile at any tau."""

import torch

from lbmtorch import BGK, D2Q9, TRT, equilibrium
from lbmtorch.macroscopic import density, velocity
from test_poiseuille import poiseuille_setup


def test_trt_equals_bgk_when_rates_match():
    torch.manual_seed(0)
    tau = 0.8
    rho = 1.0 + 0.05 * torch.rand(6, 5, dtype=torch.float64)
    u = 0.05 * (torch.rand(2, 6, 5, dtype=torch.float64) - 0.5)
    f = equilibrium(rho, u, D2Q9) * (1 + 0.01 * torch.randn(9, 6, 5, dtype=torch.float64))
    force = torch.tensor([1e-4, -2e-4], dtype=torch.float64)
    r = density(f)
    v = velocity(f, r, D2Q9, force)
    trt = TRT(tau, magic=(tau - 0.5) ** 2)  # makes tau- = tau+
    assert torch.allclose(trt(f, r, v, D2Q9, force), BGK(tau)(f, r, v, D2Q9, force), atol=1e-13)


def test_trt_poiseuille_exact_at_any_tau():
    # BGK at tau = 0.6 has ~1e-2 slip error (see test_mrt); TRT with Lambda = 3/16 has none.
    for tau in (0.6, 1.5):
        sim, u_exact = poiseuille_setup(ny_fluid=16, tau=tau)
        sim.collision = TRT(tau)
        with torch.no_grad():
            sim.run(8000 if tau < 1 else 3000)
        ux = sim.velocity[0, 0, 1:-1]
        assert (ux - u_exact).abs().max() / u_exact.max() < 1e-4, tau
