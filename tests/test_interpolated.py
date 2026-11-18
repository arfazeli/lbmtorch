"""Bouzidi bounce-back: reduces to halfway at delta = 1/2; exact Poiseuille at off-node walls."""

import torch

from lbmtorch import D2Q9, TRT, BounceBack, InterpolatedBounceBack, Simulation, viscosity_from_tau


def channel_phi(nx, ny, delta_bottom, delta_top, dtype=torch.float64):
    """Level set for a channel: walls at y = delta_bottom and y = ny - 1 - delta_top.

    Rows 0 and ny-1 are solid. delta = fluid fraction of the link from the first fluid row.
    """
    y = torch.arange(ny, dtype=dtype)
    phi = torch.minimum(y - (1 - delta_bottom), (ny - 2 + delta_top) - y)
    return phi[None, :].expand(nx, ny).clone()


def test_delta_half_equals_halfway_bounce_back():
    nx, ny = 4, 10
    phi = channel_phi(nx, ny, 0.5, 0.5)
    solid = phi < 0
    torch.manual_seed(0)
    f_post = 1 + 0.1 * torch.rand(9, nx, ny, dtype=torch.float64)
    f_stream = 1 + 0.1 * torch.rand(9, nx, ny, dtype=torch.float64)
    a = BounceBack(solid)(f_post, f_stream, D2Q9)
    b = InterpolatedBounceBack(phi)(f_post, f_stream, D2Q9)
    fluid = ~solid
    assert torch.allclose(a[:, fluid], b[:, fluid])


def test_poiseuille_with_off_node_walls():
    nx, ny, tau = 4, 18, 0.8
    db, dt = 0.2, 0.7
    phi = channel_phi(nx, ny, db, dt)
    H = (ny - 3) + db + dt  # distance between the two walls
    nu = viscosity_from_tau(tau)
    g = 0.01 * 8 * nu / H**2
    sim = Simulation(
        D2Q9,
        TRT(tau),
        (nx, ny),
        [InterpolatedBounceBack(phi)],
        force=torch.tensor([g, 0.0]),
        dtype=torch.float64,
    )
    with torch.no_grad():
        sim.run(12_000)
    y = torch.arange(1, ny - 1, dtype=torch.float64) - (1 - db)  # 0 at the bottom wall
    u_exact = g / (2 * nu) * y * (H - y)
    ux = sim.velocity[0, 0, 1:-1]
    # Measured 6e-3 on 15 fluid rows (linear Bouzidi leaves a small residual slip).
    assert (ux - u_exact).abs().max() / u_exact.max() < 1e-2


def test_gradient_wrt_wall_position():
    nx, ny = 4, 8
    phi = channel_phi(nx, ny, 0.3, 0.6).requires_grad_()
    sim = Simulation(
        D2Q9,
        TRT(0.8),
        (nx, ny),
        [InterpolatedBounceBack(phi)],
        force=torch.tensor([1e-3, 0.0]),
        dtype=torch.float64,
    )
    sim.run(20)
    (g,) = torch.autograd.grad((sim.velocity[0] ** 2).sum(), phi)
    assert torch.isfinite(g).all() and g.abs().max() > 0
