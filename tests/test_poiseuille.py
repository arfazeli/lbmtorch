"""Body-force-driven Poiseuille flow between halfway bounce-back walls."""

import torch

from lbmtorch import BGK, D2Q9, BounceBack, Simulation, viscosity_from_tau


def poiseuille_setup(ny_fluid=32, tau=0.933, u_max=0.01, dtype=torch.float64):
    nx, ny = 4, ny_fluid + 2
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    H = float(ny_fluid)  # walls sit at y = 0.5 and y = ny - 1.5, channel width = ny_fluid
    nu = viscosity_from_tau(tau)
    g = u_max * 8.0 * nu / H**2
    sim = Simulation(
        lattice=D2Q9,
        collision=BGK(tau),
        shape=(nx, ny),
        boundaries=[BounceBack(solid)],
        force=torch.tensor([g, 0.0]),
        dtype=dtype,
    )
    y = torch.arange(ny_fluid, dtype=dtype) + 0.5
    u_exact = g / (2.0 * nu) * y * (H - y)
    return sim, u_exact


def test_poiseuille_profile_matches_analytic():
    sim, u_exact = poiseuille_setup()
    with torch.no_grad():
        sim.run(20_000)
    ux = sim.velocity[0, 0, 1:-1]
    rel_err = (ux - u_exact).abs().max() / u_exact.max()
    assert rel_err < 5e-3, rel_err
    assert torch.allclose(
        sim.velocity[1].abs().max(), torch.tensor(0.0, dtype=ux.dtype), atol=1e-10
    )


def test_mass_conservation():
    sim, _ = poiseuille_setup(ny_fluid=8)
    with torch.no_grad():
        m0 = sim.f.sum()
        sim.run(200)
        assert torch.isclose(sim.f.sum(), m0)
