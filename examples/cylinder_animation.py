"""Animate vorticity in the wake of a cylinder (Schäfer-Turek 2D-2, Re = 100).

Writes ``cylinder_vorticity.gif``. Runs on the fastest available device.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from lbmtorch.benchmarks import schaefer_turek_cylinder

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
D, U = 20, 0.05
sim, cylinder = schaefer_turek_cylinder(D, U, device=device)
transient, n_frames, steps_per_frame = 40_000, 120, 100  # ~9 shedding periods of ~1330 steps
solid = cylinder.solid.cpu().numpy()
solid[:, 0] = solid[:, -1] = True  # mask the wall rows in the plot too


def vorticity():
    u = sim.velocity.cpu()
    dv_dx = torch.gradient(u[1], dim=0)[0]
    du_dy = torch.gradient(u[0], dim=1)[0]
    w = (dv_dx - du_dy).numpy() * D / U  # non-dimensional, omega D / U
    return np.where(solid, np.nan, w).T  # transpose: x horizontal, y vertical


with torch.no_grad():
    sim.run(transient)
    frames = []
    for _ in range(n_frames):
        sim.run(steps_per_frame)
        frames.append(vorticity())

fig, ax = plt.subplots(figsize=(11, 2.3), dpi=120)
im = ax.imshow(frames[0], origin="lower", cmap="RdBu_r", vmin=-4, vmax=4, interpolation="bilinear")
ax.set_axis_off()
ax.set_title(f"Cylinder wake, Re = 100: vorticity $\\omega D/U$   ({device})", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
fig.tight_layout()

anim = FuncAnimation(fig, lambda i: im.set_data(frames[i]), frames=len(frames), interval=50)
anim.save("cylinder_vorticity.gif", writer=PillowWriter(fps=20))
print("wrote cylinder_vorticity.gif")
