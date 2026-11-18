"""Lid-driven cavity at Re = 100 against Ghia, Ghia and Shin (1982), Table I."""

import numpy as np
import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation, tau_from_viscosity

# u along the vertical centreline x = 0.5, Re = 100.
GHIA_Y = np.array(
    [
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
)
GHIA_U = np.array(
    [
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
)


def cavity(n=64, u_lid=0.1, re=100.0, max_steps=20_000, tol=1e-6):
    solid = torch.zeros(n + 2, n + 2, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    wall_u = torch.zeros(2, n + 2, n + 2)
    wall_u[0, :, -1] = u_lid  # lid is the top row of solid nodes, moving in +x
    tau = tau_from_viscosity(u_lid * n / re)
    sim = Simulation(D2Q9, BGK(tau), (n + 2, n + 2), [BounceBack(solid, wall_u)])
    with torch.no_grad():
        prev = sim.velocity
        for _ in range(max_steps // 500):
            sim.run(500)
            cur = sim.velocity
            if (cur - prev).abs().max() < tol * u_lid:
                break
            prev = cur
    return sim


def test_cavity_re100_centreline():
    n, u_lid = 64, 0.1
    sim = cavity(n, u_lid)
    ux = sim.velocity[0, :, 1:-1]
    # x = 0.5 lies between fluid columns n//2 and n//2 + 1 (solid frame offsets indices by 1).
    centre = 0.5 * (ux[n // 2] + ux[n // 2 + 1]).numpy() / u_lid
    y = (np.arange(n) + 0.5) / n
    u_lbm = np.interp(GHIA_Y, y, centre)
    # Ghia's lid point is the wall itself; compare interior points only.
    err = np.abs(u_lbm[1:-1] - GHIA_U[1:-1]).max()
    # 0.006 at 40k steps; 20k steps is not fully steady, giving about 0.01.
    assert err < 0.02, err
