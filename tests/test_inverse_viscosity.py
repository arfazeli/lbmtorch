"""Recover the relaxation time from sparse velocity samples by gradient descent."""

import torch

from lbmtorch import BGK, D2Q9, Simulation
from lbmtorch.benchmarks import taylor_green
from lbmtorch.inverse import PositiveTau, observe, sparse_indices


def forward(tau, n, steps, u0, rho0):
    sim = Simulation(D2Q9, BGK(tau), (n, n), rho0=rho0, u0=u0, dtype=torch.float64)
    sim.run(steps, checkpoint_every=20)
    return sim.velocity


def test_recover_tau_from_sparse_observations():
    n, steps, tau_true = 16, 40, 0.8
    u0, rho0 = taylor_green(n, 0.02, 0.1, 0.0)
    idx = sparse_indices((n, n), 0.1)
    with torch.no_grad():
        target = observe(forward(tau_true, n, steps, u0, rho0), idx)

    tau = PositiveTau(0.65)
    opt = torch.optim.LBFGS([tau.theta], lr=1.0, max_iter=30, line_search_fn="strong_wolfe")
    losses = []

    def closure():
        opt.zero_grad()
        loss = ((observe(forward(tau, n, steps, u0, rho0), idx) - target) ** 2).sum()
        loss.backward()
        losses.append(loss.item())
        return loss

    opt.step(closure)
    assert losses[-1] < 1e-6 * losses[0], (losses[0], losses[-1])
    assert abs(float(tau) - tau_true) < 1e-3, float(tau)


def test_positive_tau_never_crosses_half():
    tau = PositiveTau(0.51)
    tau.theta.data.fill_(-50.0)
    assert float(tau) >= 0.5
