"""Zou--He boundary validation against uniform and channel-flow solutions."""

import torch

from lbmtorch import (
    BGK,
    D2Q9,
    BounceBack,
    Simulation,
    ZouHePressureOutlet,
    ZouHeVelocityInlet,
    viscosity_from_tau,
)


def test_uniform_flow_is_stationary():
    # Constant density and velocity are an exact unforced LBM solution.
    nx, ny = 12, 9
    prescribed = torch.tensor([0.04, 0.01], dtype=torch.float64)
    u0 = prescribed[:, None, None].expand(2, nx, ny)
    sim = Simulation(
        D2Q9,
        BGK(0.8),
        (nx, ny),
        boundaries=[ZouHeVelocityInlet(prescribed), ZouHePressureOutlet(1.0, uy=0.01)],
        u0=u0,
        dtype=torch.float64,
    )
    with torch.no_grad():
        sim.run(40)
    assert torch.allclose(sim.density, torch.ones(nx, ny, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(sim.velocity, u0, atol=1e-12)


def test_pressure_driven_poiseuille_profile():
    nx, ny_fluid, tau, umax = 32, 12, 0.933, 0.02
    ny = ny_fluid + 2
    nu = viscosity_from_tau(tau)
    g = 8 * nu * umax / ny_fluid**2
    y = torch.arange(ny_fluid, dtype=torch.float64) + 0.5
    profile = g * y * (ny_fluid - y) / (2 * nu)
    inlet = torch.zeros(2, ny, dtype=torch.float64)
    inlet[0, 1:-1] = profile
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    # The pressure slope dp/dx = -g is the analytic channel-flow forcing.
    rho0 = 1 + g * (nx - 1 - torch.arange(nx, dtype=torch.float64))[:, None] / D2Q9.cs2
    u0 = inlet[:, None, :].expand(2, nx, ny)
    sim = Simulation(
        D2Q9,
        BGK(tau),
        (nx, ny),
        boundaries=[ZouHeVelocityInlet(inlet), ZouHePressureOutlet(1.0), BounceBack(solid)],
        rho0=rho0,
        u0=u0,
        dtype=torch.float64,
    )
    with torch.no_grad():
        sim.run(1000)
    ux = sim.velocity[0, nx // 2, 1:-1]
    # Finite Mach number and the end boundaries introduce small profile errors.
    assert (ux - profile).abs().max() / profile.max() < 0.03


def test_gradcheck_inlet_velocity_and_outlet_density():
    def loss(inlet_speed, outlet_density):
        inlet = torch.stack((inlet_speed, torch.zeros_like(inlet_speed)))
        sim = Simulation(
            D2Q9,
            BGK(0.8),
            (5, 4),
            boundaries=[ZouHeVelocityInlet(inlet), ZouHePressureOutlet(outlet_density)],
            dtype=torch.float64,
        )
        sim.run(3)
        return (sim.velocity**2).sum()

    speed = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    rho = torch.tensor(1.01, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(loss, (speed, rho), eps=1e-6, atol=1e-8)
