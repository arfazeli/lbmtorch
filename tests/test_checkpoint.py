"""Gradient checkpointing must not change gradients, only memory."""

import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation


def grad_of_ke(tau, checkpoint_every):
    nx, ny = 6, 8
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    sim = Simulation(
        D2Q9,
        BGK(tau),
        (nx, ny),
        [BounceBack(solid)],
        force=torch.tensor([1e-3, 0.0], dtype=torch.float64),
        dtype=torch.float64,
    )
    sim.run(23, checkpoint_every=checkpoint_every)
    assert sim.step_count == 23
    (g,) = torch.autograd.grad((sim.velocity**2).sum(), tau)
    return g


def test_checkpointed_gradient_matches_plain():
    tau = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    g_plain = grad_of_ke(tau, None)
    g_ckpt = grad_of_ke(tau, 5)
    assert torch.allclose(g_plain, g_ckpt, rtol=1e-12), (g_plain, g_ckpt)


def test_checkpoint_ignored_without_grad():
    sim = Simulation(D2Q9, BGK(0.8), (4, 4))
    with torch.no_grad():
        sim.run(7, checkpoint_every=3)
    assert sim.step_count == 7
