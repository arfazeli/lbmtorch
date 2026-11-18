"""Poiseuille flow: run, compare with the analytic profile, and plot."""

import matplotlib.pyplot as plt
import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation, viscosity_from_tau

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
nx, ny_fluid, tau, u_max = 8, 64, 0.933, 0.01
ny = ny_fluid + 2

solid = torch.zeros(nx, ny, dtype=torch.bool)
solid[:, 0] = solid[:, -1] = True
nu = viscosity_from_tau(tau)
g = u_max * 8 * nu / ny_fluid**2

sim = Simulation(
    D2Q9, BGK(tau), (nx, ny), [BounceBack(solid)], force=torch.tensor([g, 0.0]), device=device
)
with torch.no_grad():
    sim.run(40_000)

y = torch.arange(ny_fluid) + 0.5
u_exact = g / (2 * nu) * y * (ny_fluid - y)
ux = sim.velocity[0, 0, 1:-1].cpu()
print(f"device={device}  max rel error = {((ux - u_exact).abs().max() / u_exact.max()):.2e}")

plt.plot(y, u_exact, "k-", label="analytic")
plt.plot(y, ux, "o", mfc="none", label="lbmtorch")
plt.xlabel("y (lattice units)")
plt.ylabel("u_x")
plt.legend()
plt.savefig("poiseuille.png", dpi=150)
print("wrote poiseuille.png")
