"""Lid-driven cavity at Re = 100: streamlines and speed, compared with Ghia et al. (1982)."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation, tau_from_viscosity

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
n, u_lid, re = 128, 0.1, 100.0

solid = torch.zeros(n + 2, n + 2, dtype=torch.bool)
solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
wall_u = torch.zeros(2, n + 2, n + 2)
wall_u[0, :, -1] = u_lid
sim = Simulation(
    D2Q9,
    BGK(tau_from_viscosity(u_lid * n / re)),
    (n + 2, n + 2),
    [BounceBack(solid, wall_u)],
    device=device,
)
with torch.no_grad():
    sim.run(60_000)

u = sim.velocity[:, 1:-1, 1:-1].cpu().numpy() / u_lid  # fluid nodes only, normalised by lid speed
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="ij")
speed = np.hypot(u[0], u[1])

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 4.5), dpi=120, gridspec_kw={"width_ratios": [1.15, 1]}
)
im = ax1.pcolormesh(X, Y, speed, cmap="viridis", shading="auto")
ax1.streamplot(x, x, u[0].T, u[1].T, color="w", linewidth=0.6, density=1.4, arrowsize=0.7)
ax1.set_aspect("equal")
ax1.set_title(f"Lid-driven cavity, Re = {re:.0f}: |u|/U and streamlines")
fig.colorbar(im, ax=ax1, fraction=0.046)

ghia_y = [
    1.0,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0,
]
ghia_u = [
    1.0,
    0.84123,
    0.78871,
    0.73722,
    0.68717,
    0.23151,
    0.00332,
    -0.13641,
    -0.20581,
    -0.2109,
    -0.15662,
    -0.1015,
    -0.06434,
    -0.04775,
    -0.04192,
    -0.03717,
    0.0,
]
centre = 0.5 * (u[0, n // 2 - 1] + u[0, n // 2])
ax2.plot(centre, x, "-", label="lbmtorch")
ax2.plot(ghia_u, ghia_y, "ko", mfc="none", label="Ghia et al. (1982)")
ax2.set_xlabel("$u_x / U$ on $x = 0.5$")
ax2.set_ylabel("$y$")
ax2.legend()
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("cavity.png")
print("wrote cavity.png")
