"""Recover the fluid viscosity from sparse velocity samples by differentiating through the solver.

A Taylor-Green vortex is simulated with an unknown relaxation time. We observe the velocity
at 10% of the nodes after 60 steps and fit tau with L-BFGS. Writes ``inverse_viscosity.png``.
"""

import matplotlib.pyplot as plt
import torch

from lbmtorch import BGK, D2Q9, Simulation
from lbmtorch.benchmarks import taylor_green
from lbmtorch.inverse import PositiveTau, observe, sparse_indices
from lbmtorch.units import viscosity_from_tau

n, steps, tau_true, tau_start = 32, 60, 0.8, 0.6
u0, rho0 = taylor_green(n, 0.02, viscosity_from_tau(tau_true), 0.0)
idx = sparse_indices((n, n), 0.1)


def forward(tau):
    sim = Simulation(D2Q9, BGK(tau), (n, n), rho0=rho0, u0=u0, dtype=torch.float64)
    sim.run(steps, checkpoint_every=20)
    return sim.velocity


with torch.no_grad():
    target = observe(forward(tau_true), idx)
    target = target + 1e-4 * torch.randn_like(target)  # 0.5% measurement noise

tau = PositiveTau(tau_start)
opt = torch.optim.LBFGS([tau.theta], lr=1.0, max_iter=40, line_search_fn="strong_wolfe")
history = []


def closure():
    opt.zero_grad()
    loss = ((observe(forward(tau), idx) - target) ** 2).sum()
    loss.backward()
    history.append((float(tau), loss.item()))
    return loss


opt.step(closure)
print(f"true tau {tau_true}, recovered {float(tau):.5f} after {len(history)} evaluations")
print(
    f"true nu  {viscosity_from_tau(tau_true):.5f}, recovered {viscosity_from_tau(float(tau)):.5f}"
)

taus = torch.linspace(0.55, 1.1, 40)
with torch.no_grad():
    landscape = [((observe(forward(t.item()), idx) - target) ** 2).sum().item() for t in taus]

fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
ax.semilogy(taus, landscape, "-", color="0.6", label="loss landscape")
ax.semilogy([h[0] for h in history], [h[1] for h in history], "o-", ms=4, label="L-BFGS iterates")
ax.axvline(tau_true, color="k", ls="--", lw=0.8, label="true $\\tau$")
ax.set_xlabel("$\\tau$")
ax.set_ylabel("misfit")
ax.set_title(f"Viscosity inversion from {len(idx)} of {n * n} nodes, {steps} steps")
ax.legend()
fig.tight_layout()
fig.savefig("inverse_viscosity.png")
print("wrote inverse_viscosity.png")
