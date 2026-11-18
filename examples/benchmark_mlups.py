"""Throughput in million lattice updates per second across devices and torch.compile.

Lid-driven-cavity-style setup (BGK + one bounce-back object) so all devices run the same
kernels. Prints a table; the numbers feed the performance section of the docs.
"""

import time

import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation


def build(n, device, dtype):
    solid = torch.zeros(n, n, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    wall_u = torch.zeros(2, n, n)
    wall_u[0, :, -1] = 0.05
    return Simulation(
        D2Q9, BGK(0.8), (n, n), [BounceBack(solid, wall_u)], device=device, dtype=dtype
    )


def mlups(sim, step_fn, steps=200, warmup=20):
    f = sim.f
    for _ in range(warmup):
        f = step_fn(f)
    sync(sim.device)
    t0 = time.perf_counter()
    for _ in range(steps):
        f = step_fn(f)
    sync(sim.device)
    return steps * f[0].numel() / (time.perf_counter() - t0) / 1e6


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


devices = (
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else [])
)
print(f"{'grid':>8} {'device':>6} {'dtype':>8} {'eager MLUPS':>12} {'compiled MLUPS':>15}")
with torch.no_grad():
    for n in (64, 128, 256, 512):
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                if device == "mps" and dtype == torch.float64:
                    continue
                sim = build(n, device, dtype)
                eager = mlups(sim, sim.advance)
                try:
                    compiled = torch.compile(sim.advance)
                    comp = mlups(sim, compiled)
                except Exception as exc:  # report and move on
                    comp = float("nan")
                    print(f"  (compile failed on {device}: {type(exc).__name__})")
                print(f"{n:>5}x{n:<3} {device:>6} {str(dtype)[6:]:>8} {eager:>12.1f} {comp:>15.1f}")
