"""Schäfer-Turek 2D-2: Strouhal number and force coefficients. Slow; run with -m slow."""

import pytest
import torch

from lbmtorch.benchmarks import force_coefficients, schaefer_turek_cylinder, strouhal_from_signal


@pytest.mark.slow
def test_cylinder_re100_strouhal_and_forces():
    D, U = 20, 0.05
    sim, cylinder = schaefer_turek_cylinder(D, U)
    every = 20
    cd, cl = [], []
    with torch.no_grad():
        sim.run(60_000)
        for _ in range(1000):
            sim.run(every)
            f_post = sim.collision(sim.f, sim.density, sim.velocity, sim.lattice, sim.force)
            c_d, c_l = force_coefficients(cylinder.force(f_post, sim.lattice), U, D)
            cd.append(c_d)
            cl.append(c_l)
    st = strouhal_from_signal(cl, every, U, D)
    # Measured with D = 20 and Bouzidi walls: St 0.302, Cd 3.25, Cl_max 0.94
    # (staircase bounce-back gives 0.296, 3.52, 1.21).
    assert 0.29 < st < 0.31, st
    assert 3.1 < sum(cd) / len(cd) < 3.4, sum(cd) / len(cd)
    assert 0.85 < max(cl) < 1.1, max(cl)
