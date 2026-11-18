"""Inverse design: reshape an obstacle to minimise drag at fixed area.

An ellipse in a body-force-driven channel. Parameter: log aspect ratio a/b at fixed area.
Gradients of the momentum-exchange drag flow through the Bouzidi boundary's level set.
Writes ``inverse_design.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from lbmtorch import D2Q9, TRT, BounceBack, InterpolatedBounceBack, Simulation
from lbmtorch.geometry import channel_walls, ellipse

NX, NY, TAU, G, STEPS = 60, 30, 0.8, 2e-5, 600
CENTER, AREA_R = (20.0, 14.5), 5.0


def drag_and_field(log_s):
    s = torch.exp(log_s)
    a, b = AREA_R * torch.sqrt(s), AREA_R / torch.sqrt(s)
    body = InterpolatedBounceBack(ellipse((NX, NY), CENTER, a, b))
    sim = Simulation(
        D2Q9,
        TRT(TAU),
        (NX, NY),
        [BounceBack(channel_walls((NX, NY))), body],
        force=torch.tensor([G, 0.0]),
        dtype=torch.float64,
    )
    sim.run(STEPS, checkpoint_every=50)
    f_post = sim.collision(sim.f, sim.density, sim.velocity, sim.lattice, sim.force)
    return body.force(f_post, sim.lattice)[0], sim.velocity.detach(), body.solid


log_s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)  # start from a circle
opt = torch.optim.Adam([log_s], lr=0.1)
history = []
for it in range(12):
    opt.zero_grad()
    drag, u, solid = drag_and_field(log_s)
    drag.backward()
    history.append((np.exp(log_s.item()), drag.item()))
    if it == 0:
        u_start, solid_start = u, solid
    print(f"iter {it:2d}  aspect a/b = {history[-1][0]:.3f}  drag = {drag.item():.4e}")
    opt.step()
    with torch.no_grad():
        log_s.clamp_(max=np.log(2.4))  # keep the body inside the channel

fig, axes = plt.subplots(
    1, 3, figsize=(13, 3.2), dpi=120, gridspec_kw={"width_ratios": [1.6, 1.6, 1]}
)
for ax, (uu, ss), title in zip(
    axes[:2], [(u_start, solid_start), (u, solid)], ["start: circle", "optimised"], strict=True
):
    speed = np.hypot(uu[0].numpy(), uu[1].numpy())
    speed[ss.numpy()] = np.nan
    ax.imshow(speed.T, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_axis_off()
axes[2].plot([h[0] for h in history], [h[1] / history[0][1] for h in history], "o-", ms=4)
axes[2].set_xlabel("aspect ratio a/b")
axes[2].set_ylabel("drag / initial drag")
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("inverse_design.png")
print("wrote inverse_design.png")
