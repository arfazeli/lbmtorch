import torch

from lbmtorch import D2Q9, density, equilibrium, velocity


def test_equilibrium_recovers_moments():
    torch.manual_seed(0)
    rho = 1.0 + 0.1 * torch.rand(5, 7, dtype=torch.float64)
    u = 0.05 * (torch.rand(2, 5, 7, dtype=torch.float64) - 0.5)
    feq = equilibrium(rho, u, D2Q9)
    assert torch.allclose(density(feq), rho)
    assert torch.allclose(velocity(feq, rho, D2Q9), u)
