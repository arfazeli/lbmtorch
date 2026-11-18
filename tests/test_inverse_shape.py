"""Shape inversion and inverse design through the differentiable level set."""

import torch

from lbmtorch import D2Q9, TRT, BounceBack, InterpolatedBounceBack, Simulation
from lbmtorch.geometry import channel_walls, circle, ellipse
from lbmtorch.inverse import observe, sparse_indices

NX, NY, TAU, G, STEPS = 40, 22, 0.8, 2e-5, 300
CENTER = (14.0, 10.5)


def channel_flow(phi):
    sim = Simulation(
        D2Q9,
        TRT(TAU),
        (NX, NY),
        [BounceBack(channel_walls((NX, NY))), InterpolatedBounceBack(phi)],
        force=torch.tensor([G, 0.0]),
        dtype=torch.float64,
    )
    sim.run(STEPS, checkpoint_every=50)
    return sim


def test_recover_radius_from_velocity_samples():
    idx = sparse_indices((NX, NY), 0.2)
    with torch.no_grad():
        target = observe(channel_flow(circle((NX, NY), CENTER, 4.0)).velocity, idx)

    r = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([r], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (
            (observe(channel_flow(circle((NX, NY), CENTER, r)).velocity, idx) - target) ** 2
        ).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert abs(r.item() - 4.0) < 0.1, r.item()


def test_streamlining_reduces_drag():
    # Ellipse of fixed area pi*a*b; aspect s = a/b. Drag should fall as s grows from 1.
    area_r = 4.0
    log_s = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    s = torch.exp(log_s)
    a, b = area_r * torch.sqrt(s), area_r / torch.sqrt(s)
    phi = ellipse((NX, NY), CENTER, a, b)
    sim = channel_flow(phi)
    body = sim.boundaries[1]
    f_post = sim.collision(sim.f, sim.density, sim.velocity, sim.lattice, sim.force)
    drag = body.force(f_post, sim.lattice)[0]
    (g,) = torch.autograd.grad(drag, log_s)
    assert drag > 0
    assert g < 0, g  # elongating along the flow lowers drag
