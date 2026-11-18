# lbmtorch

A differentiable lattice Boltzmann solver for 2D incompressible flow, written in PyTorch.

Every step of the time loop is a plain tensor operation, so the solver runs on CPU, CUDA or
Apple MPS, and `torch.autograd` gives gradients of any output with respect to viscosity, body
force, initial state or wall geometry. That makes it usable as a forward model for inverse
problems, shape optimisation and training learned closures with the solver in the loop.

## What's included

- D2Q9 lattice; BGK, TRT, MRT and Smagorinsky LES collision; Guo body forcing
- Bounce-back walls (static or moving), Bouzidi curved walls from a level set,
  Zou-He velocity inlet and pressure outlet, momentum-exchange force on bodies
- Gradient checkpointing for long differentiable runs
- Validated against Poiseuille flow, Taylor-Green decay (second-order convergence),
  the lid-driven cavity (Ghia et al. 1982) and the Schäfer-Turek cylinder benchmark

## Install

```bash
pip install -e .            # library
pip install -e .[examples]  # plus matplotlib for the example scripts
pip install -e .[dev]       # plus tests and lint
```

## Usage

A simulation is a lattice, a collision operator and a list of boundary conditions.
Force-driven channel flow:

```python
import torch
from lbmtorch import BGK, D2Q9, BounceBack, Simulation

nx, ny = 4, 34
walls = torch.zeros(nx, ny, dtype=torch.bool)
walls[:, 0] = walls[:, -1] = True

sim = Simulation(
    lattice=D2Q9,
    collision=BGK(tau=1.0),
    shape=(nx, ny),
    boundaries=[BounceBack(walls)],
    force=torch.tensor([1e-5, 0.0]),
)
sim.run(10_000)
u = sim.velocity  # shape (2, nx, ny): a parabolic profile across the channel
```

Gradients flow through the whole run. Make a parameter a tensor with `requires_grad`:

```python
tau = torch.tensor(0.8, requires_grad=True)
sim = Simulation(
    lattice=D2Q9,
    collision=BGK(tau=tau),
    shape=(nx, ny),
    boundaries=[BounceBack(walls)],
    force=torch.tensor([1e-5, 0.0]),
)
sim.run(500, checkpoint_every=100)
sim.velocity[0].mean().backward()
print(tau.grad)
```

Pass `device="cuda"` or `device="mps"` to `Simulation` to run on a GPU.

## Examples

| Script | Shows |
|---|---|
| `examples/poiseuille.py` | Channel flow against the analytic profile |
| `examples/cavity.py` | Lid-driven cavity against Ghia et al. |
| `examples/cylinder.py`, `cylinder_animation.py` | Vortex shedding, Strouhal number, drag and lift |
| `examples/inverse_viscosity.py` | Recover viscosity from sparse velocity samples |
| `examples/inverse_design.py` | Reshape an obstacle for minimum drag |
| `examples/learned_collision.py` | Train a neural collision correction through the solver |
| `examples/benchmark_mlups.py` | Throughput on CPU, CUDA, MPS and `torch.compile` |

Equations and references are in [`docs/theory.md`](docs/theory.md); a longer walkthrough of the
design is in [`docs/learning_guide.md`](docs/learning_guide.md).

## Extending

A collision operator is any callable `(f, rho, u, lattice, force=None) -> f_post`; a
boundary condition is any callable `(f_post, f_stream, lattice) -> f_new`. Build results
with `torch.where` and `torch.stack` rather than in-place writes so gradients stay defined.

## Tests

```bash
pytest            # fast suite
pytest -m slow    # cylinder benchmark, several minutes
```

## License

MIT. See [`LICENSE.md`](LICENSE.md).
