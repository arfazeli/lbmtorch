"""Learned MRT: exact MRT at init, exact conservation, error reduction after training."""

import torch

from lbmtorch import D2Q9, MRT, Simulation, equilibrium
from lbmtorch.benchmarks import taylor_green
from lbmtorch.learned import LearnedMRT
from lbmtorch.macroscopic import density, momentum, velocity
from lbmtorch.units import viscosity_from_tau

N, TAU, STEPS = 16, 0.6, 30
NU = viscosity_from_tau(TAU)


def trajectory_error(collision, amp, shift=0):
    u0, rho0 = taylor_green(N, amp, NU, 0.0)
    u0, rho0 = torch.roll(u0, shift, dims=(1,)), torch.roll(rho0, shift, dims=0)
    sim = Simulation(D2Q9, collision, (N, N), rho0=rho0, u0=u0, dtype=torch.float64)
    err = 0.0
    for t in range(1, STEPS + 1):
        sim.step()
        if t % 10 == 0:
            ue, _ = taylor_green(N, amp, NU, t)
            ue = torch.roll(ue, shift, dims=(1,))
            err = err + ((sim.velocity - ue) ** 2).sum() / (ue**2).sum()
    return err / (STEPS // 10)


def test_untrained_model_is_exactly_mrt():
    torch.manual_seed(0)
    rho = 1 + 0.02 * torch.rand(6, 6, dtype=torch.float64)
    u = 0.05 * torch.rand(2, 6, 6, dtype=torch.float64)
    f = equilibrium(rho, u, D2Q9) * (1 + 0.01 * torch.randn(9, 6, 6, dtype=torch.float64))
    model = LearnedMRT(TAU).double()
    assert torch.allclose(model(f, rho, u, D2Q9), MRT(TAU)(f, rho, u, D2Q9))


def test_conservation_with_random_weights():
    torch.manual_seed(1)
    model = LearnedMRT(TAU).double()
    for p in model.parameters():
        p.data.normal_()
    rho0 = 1 + 0.02 * torch.rand(6, 6, dtype=torch.float64)
    u0 = 0.05 * torch.rand(2, 6, 6, dtype=torch.float64)
    f = equilibrium(rho0, u0, D2Q9) * (1 + 0.01 * torch.randn(9, 6, 6, dtype=torch.float64))
    rho, u = density(f), velocity(f, density(f), D2Q9)  # moments of the actual field
    f_post = model(f, rho, u, D2Q9)
    assert torch.allclose(density(f_post), rho)
    assert torch.allclose(momentum(f_post, D2Q9), momentum(f, D2Q9))


def test_training_reduces_error_on_held_out_case():
    torch.manual_seed(0)
    model = LearnedMRT(TAU).double()
    with torch.no_grad():
        base_train = trajectory_error(MRT(TAU), 0.08).item()
        base_test = trajectory_error(MRT(TAU), 0.06, shift=4).item()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(100):
        opt.zero_grad()
        loss = trajectory_error(model, 0.08)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train = trajectory_error(model, 0.08).item()
        test = trajectory_error(model, 0.06, shift=4).item()
    # examples/learned_collision.py (150 iterations): 5.7e-4 -> 8.9e-5 train,
    # 5.8e-4 -> 9.3e-5 held-out.
    assert train < base_train / 3, (base_train, train)
    assert test < base_test / 3, (base_test, test)
