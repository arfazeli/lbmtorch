"""Train a learned collision correction solver-in-the-loop and test it on unseen flows.

Reference: analytic Taylor-Green solution at coarse resolution (16 x 16) and moderate Mach
number, where compressibility error is visible. Writes ``learned_collision.png``.
"""

import matplotlib.pyplot as plt
import torch

from lbmtorch import D2Q9, MRT, Simulation
from lbmtorch.benchmarks import taylor_green
from lbmtorch.learned import LearnedMRT
from lbmtorch.units import viscosity_from_tau

torch.manual_seed(0)
N, TAU, STEPS = 16, 0.6, 30
NU = viscosity_from_tau(TAU)


def error_curve(collision, amp, shift=0, steps=STEPS):
    u0, rho0 = taylor_green(N, amp, NU, 0.0)
    u0, rho0 = torch.roll(u0, shift, dims=(1,)), torch.roll(rho0, shift, dims=0)
    sim = Simulation(D2Q9, collision, (N, N), rho0=rho0, u0=u0, dtype=torch.float64)
    errs = []
    for t in range(1, steps + 1):
        sim.step()
        ue, _ = taylor_green(N, amp, NU, t)
        ue = torch.roll(ue, shift, dims=(1,))
        errs.append(((sim.velocity - ue) ** 2).sum() / (ue**2).sum())
    return torch.stack(errs)


model = LearnedMRT(TAU).double()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
losses = []
for it in range(150):
    opt.zero_grad()
    loss = error_curve(model, 0.08)[9::10].mean()  # match at t = 10, 20, 30
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if it % 30 == 0:
        print(f"iter {it:3d}  loss {loss.item():.3e}")

cases = {
    "train: amp 0.08": (0.08, 0),
    "held-out: amp 0.06, shifted": (0.06, 4),
    "held-out: 2x longer": (0.08, 0),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=120)
axes[0].semilogy(losses)
axes[0].set_xlabel("training iteration")
axes[0].set_ylabel("loss (relative velocity error$^2$)")
axes[0].set_title("Solver-in-the-loop training")
with torch.no_grad():
    for (label, (amp, shift)), color in zip(cases.items(), ["C0", "C1", "C2"], strict=True):
        steps = 2 * STEPS if "longer" in label else STEPS
        base = error_curve(MRT(TAU), amp, shift, steps).sqrt()
        learned = error_curve(model, amp, shift, steps).sqrt()
        axes[1].plot(base, "--", color=color, label=f"MRT, {label}")
        axes[1].plot(learned, "-", color=color, label=f"learned, {label}")
        print(f"{label:30s} final rel. error: MRT {base[-1]:.3e}  learned {learned[-1]:.3e}")
axes[1].set_xlabel("time step")
axes[1].set_ylabel("relative velocity error")
axes[1].set_title("Generalisation")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("learned_collision.png")
print("wrote learned_collision.png")
