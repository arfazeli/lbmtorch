"""Momentum-exchange force: Poiseuille wall shear must balance the body force."""

import torch

from lbmtorch import BounceBack
from test_poiseuille import poiseuille_setup


def test_wall_force_balances_body_force():
    sim, _ = poiseuille_setup(ny_fluid=16, tau=0.933)
    with torch.no_grad():
        sim.run(8000)
        f_post = sim.collision(sim.f, sim.density, sim.velocity, sim.lattice, sim.force)
    walls: BounceBack = sim.boundaries[0]
    force = walls.force(f_post, sim.lattice)
    n_fluid = (~walls.solid).sum()
    expected = sim.force[0] * n_fluid  # rho = 1, total body force on the fluid
    assert torch.isclose(force[0], expected, rtol=1e-3), (force, expected)
    assert abs(force[1]) < 1e-10 * expected
