"""Gradients through the time loop must match finite differences."""

import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation


def kinetic_energy_after(tau, g, n_steps=5):
    nx, ny = 4, 8
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    sim = Simulation(
        lattice=D2Q9,
        collision=BGK(tau),
        shape=(nx, ny),
        boundaries=[BounceBack(solid)],
        force=torch.stack([g, torch.zeros_like(g)]),
        dtype=torch.float64,
    )
    sim.run(n_steps)
    return 0.5 * (sim.velocity[:, :, 1:-1] ** 2).sum()


def test_gradcheck_tau_and_force():
    tau = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(1e-3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(kinetic_energy_after, (tau, g), eps=1e-6, atol=1e-8)


def test_gradient_wrt_initial_velocity():
    nx, ny = 6, 6
    u0 = (1e-3 * torch.randn(2, nx, ny, dtype=torch.float64)).requires_grad_()
    sim = Simulation(D2Q9, BGK(0.7), (nx, ny), u0=u0, dtype=torch.float64)
    sim.run(3)
    loss = (sim.velocity**2).sum()
    loss.backward()
    assert u0.grad is not None and torch.isfinite(u0.grad).all()
