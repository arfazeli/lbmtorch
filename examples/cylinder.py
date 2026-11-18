"""Vortex shedding behind a cylinder: Schäfer-Turek 2D-2 benchmark at Re = 100."""

import time

import torch

from lbmtorch.benchmarks import force_coefficients, schaefer_turek_cylinder, strouhal_from_signal

device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
D, U = 20, 0.05
sim, cylinder = schaefer_turek_cylinder(D, U, device=device)
sample_every, transient, measure = 20, 60_000, 20_000  # ~15 shedding periods of ~1330 steps

cd, cl = [], []
t0 = time.time()
with torch.no_grad():
    sim.run(transient)
    for _ in range(measure // sample_every):
        sim.run(sample_every)
        f_post = sim.collision(sim.f, sim.density, sim.velocity, sim.lattice, sim.force)
        c_d, c_l = force_coefficients(cylinder.force(f_post, sim.lattice), U, D)
        cd.append(c_d)
        cl.append(c_l)
elapsed = time.time() - t0
st = strouhal_from_signal(cl, sample_every, U, D)
print(
    f"device={device}  {sim.step_count} steps in {elapsed:.0f}s "
    f"({sim.step_count * sim.f[0].numel() / elapsed / 1e6:.1f} MLUPS)"
)
print(
    f"St={st:.3f} (ref 0.295-0.305)  Cd_mean={sum(cd) / len(cd):.3f} (ref 3.22-3.24)  "
    f"Cl_max={max(cl):.3f} (ref 0.99-1.01)"
)
