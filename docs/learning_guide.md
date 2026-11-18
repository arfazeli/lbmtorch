# lbmtorch learning guide

A study companion to the code. It follows the order in which the pieces were built and
explains *why* each one looks the way it does. Read it next to the source; every section
names the file it describes. `docs/theory.md` holds the equations; this file holds the
reasoning. Sections marked **Self-check** are questions to answer before moving on.

---

## 1. What the lattice Boltzmann method is, in one page

Navier-Stokes solvers discretise the macroscopic equations for velocity and pressure.
LBM instead evolves a small set of *populations* $f_q(\mathbf{x}, t)$, one per discrete
velocity $\mathbf{c}_q$, on a regular grid. Each time step has two phases:

1. **Collision** (local): populations at a node relax toward a Maxwell-Boltzmann
   equilibrium that depends only on the local density and velocity.
2. **Streaming** (non-local, but trivial): each population hops to the neighbouring node
   in its direction, $f_q(\mathbf{x} + \mathbf{c}_q, t+1) = f_q^\star(\mathbf{x}, t)$.

Density and momentum are sums over the populations. A Chapman-Enskog expansion shows the
scheme recovers the weakly compressible Navier-Stokes equations with viscosity
$\nu = c_s^2(\tau - 1/2)$, where $\tau$ is the relaxation time.

Why it matters for this project:

- **Everything is local or a shift.** Collision is a pointwise tensor op; streaming is
  `torch.roll`. No linear solves, no sparse matrices, no mesh. This is why LBM maps so well
  to GPUs and why it can be made differentiable with autograd for free.
- **Pressure is explicit.** $p = c_s^2 \rho$, so there is no Poisson solve for pressure.
  The price is weak compressibility: velocities must stay below roughly $0.1\,c_s$.
- **Boundaries are population rules,** not matrix rows. Bounce-back is one `torch.where`.

**Self-check.** Why must $\tau > 1/2$? What happens to $\nu$ as $\tau \to 1/2$?

---

## 2. Repository layout and the reasoning behind it

```
src/lbmtorch/
  lattice/       velocity set as data (c, w, opp, cs2)
  macroscopic.py moments and equilibrium: the only place physics formulas live
  collision/     BGK, TRT, MRT, Smagorinsky: all satisfy one Protocol
  boundary/      BounceBack, InterpolatedBounceBack, Zou-He: all satisfy one Protocol
  simulation.py  stream() and the Simulation driver
  units.py       tau <-> viscosity
  benchmarks.py  reference setups (Taylor-Green, cylinder) shared by examples and tests
  inverse.py     bounded tau parameterisation, sparse observations
  geometry.py    differentiable level sets (circle, ellipse), channel walls
  learned.py     LearnedMRT: MRT plus a conservative neural correction
```

**src layout.** The package lives under `src/` so that tests import the *installed*
package, never the working directory by accident. It is the layout most Python
packaging guides now recommend.

**Protocols instead of base classes.** `CollisionOperator` and `BoundaryCondition` are
`typing.Protocol`s: any callable with the right signature qualifies. There is no registry
and no inheritance. `LearnedMRT` (section 15.3) is a plain `torch.nn.Module`
whose `forward` has the same signature. This is "duck typing with a type checker".

**Population layout `(Q, *spatial)`.** Velocity index first means `f[q]` is a spatial field
and `f.sum(0)` is density. It also means `torch.roll(f[q], ...)` streams one direction with
no reshaping.

**Self-check.** Open `collision/base.py`. What would you have to change to add a
D3Q19 collision operator? (Answer: nothing there. The Protocol is dimension-agnostic.)

---

## 3. The lattice (`lattice/base.py`, `lattice/d2q9.py`)

A `Lattice` is a frozen dataclass holding four tensors and $c_s^2$. `opp[q]` is computed
rather than typed by hand: `opposite_indices` matches each $\mathbf{c}_q$ with
$-\mathbf{c}_q$ and raises if the set is not symmetric. That single check would have
caught a typo in the velocity table before any physics test ran.

Weights are stored in float64 and cast per simulation (`Lattice.to`). Storing them in
float32 and upcasting would lose precision for float64 gradient checks.

The isotropy test in `tests/test_lattice.py` verifies
$\sum_q w_q c_{q\alpha} c_{q\beta} = c_s^2 \delta_{\alpha\beta}$. That identity is what
makes the recovered pressure isotropic; it is the first thing to check on any new lattice.

---

## 4. Moments and equilibrium (`macroscopic.py`)

Three functions do all the physics bookkeeping:

- `density`: $\rho = \sum_q f_q$
- `momentum`: $\rho\mathbf{u} = \sum_q \mathbf{c}_q f_q$, written as one `einsum`
  with an ellipsis so it works in any dimension.
- `velocity`: momentum over density, **plus half the body force**. This half-force shift
  is part of the Guo forcing scheme (section 6). Forgetting it produces a velocity that is
  off by $F/2\rho$, which is a first-order error in the force.

`equilibrium` is the second-order expansion. The `w_bcast` helper reshapes weights to
`(Q, 1, 1)` so broadcasting works without hard-coding two dimensions.

`tests/test_macroscopic.py` checks that the moments of `equilibrium(rho, u)` return
`rho` and `u` exactly. This is the minimal consistency property of any equilibrium.

**Self-check.** The equilibrium is second order in $u$. What symptom appears if you push
$|u|$ to 0.3 in lattice units?

---

## 5. The time loop (`simulation.py`)

`stream` is nine `torch.roll` calls stacked. `torch.roll(x, 1)` gives `out[i] = in[i-1]`,
so rolling by $+\mathbf{c}_q$ moves the population from $\mathbf{x}$ to
$\mathbf{x} + \mathbf{c}_q$. Periodicity comes free.

`Simulation.advance(f, n)` is the pure function
$f \mapsto$ (collide, stream, boundaries)$^n(f)$. `step` and `run` are thin wrappers that
update `self.f`. Keeping the pure function separate is what makes gradient checkpointing
(section 12) a three-line addition.

A bug worth remembering from the first day: after indexing `f[q]`, the spatial axes start
at 0, not 1. The first version rolled along dims `(1, 2)` of a 2-D tensor and crashed.
Tests caught it immediately; that is what they are for.

---

## 6. BGK collision and Guo forcing (`collision/bgk.py`)

BGK relaxes every population toward equilibrium at the same rate $1/\tau$. Body forces
enter through Guo's scheme, which has two halves that must be used together:

1. the source term $S_q$ added to the post-collision populations, scaled by
   $(1 - 1/2\tau)$;
2. the half-force shift in the velocity definition (section 4).

Together they cancel the discrete lattice error that a naive forcing produces. The Guo
paper (2002) is short and worth reading once; the derivation is a Chapman-Enskog expansion
with the force included.

`tau` may be a tensor with `requires_grad=True`. Nothing in the code converts it to a
float, so gradients with respect to viscosity flow through every step.

**Self-check.** In `BGK.__call__`, why is the force term multiplied by
$(1 - \omega/2)$ rather than 1?

---

## 7. Halfway bounce-back (`boundary/bounce_back.py`)

The idea: a population about to stream from fluid node $\mathbf{x}$ into a solid node
$\mathbf{x} + \mathbf{c}_q$ is instead sent back to $\mathbf{x}$ with reversed direction.
The wall is effectively halfway between the two nodes, which is why a channel with
`ny_fluid` fluid rows has width exactly `ny_fluid`.

Implementation details that matter:

- `hit = roll(solid, -c_q)` marks fluid nodes whose neighbour in direction $q$ is solid.
  The roll is periodic, so a solid row at the bottom also "hits" from the top row: this
  is correct for bounce-back (any link into solid is reflected) but wrong for force
  evaluation (section 9).
- Results are built with `torch.where` on a list and one `torch.stack`. In-place
  assignment into a tensor autograd still needs would raise or silently corrupt gradients.
- **Solid nodes keep finite junk.** Zeroing them would make `rho = 0` there, `u = NaN`,
  and `torch.where(mask, good, NaN)` has a NaN *gradient* even where the mask selects the
  good branch. This is the single most common autograd trap in masked physics code.
- **Moving walls** (Ladd 1994) subtract $2 w_q \rho\, \mathbf{c}_q\cdot\mathbf{u}_w / c_s^2$.
  The lid-driven cavity is just a stationary box whose top solid row carries
  $\mathbf{u}_w = (U, 0)$.

The wall location is exactly midway only at $(\tau - 1/2)^2 = 3/16$, i.e. $\tau \approx
0.933$. The Poiseuille test uses that value so the tolerance can be tight. At other
$\tau$ there is a small slip; MRT/TRT collision with the right "magic" parameter fixes it.

**Self-check.** Sketch the three populations at the top fluid row that get reflected, and
which populations they become.

---

## 7b. Curved walls: interpolated bounce-back (`boundary/interpolated.py`)

Halfway bounce-back puts every wall midway between nodes, so a circle becomes a
staircase. Bouzidi's scheme fixes this with one extra number per link: $\delta$, the
fraction of the link that lies in the fluid. It comes from a level set $\phi$ by linear
interpolation between the fluid and solid node values. The reflected population is then
a linear combination of two post-collision populations, chosen so the effective no-slip
point lands at $\delta$ rather than at $1/2$.

Three things to notice in the code:

- The whole boundary is a *function of $\phi$*, a tensor. Autograd therefore gives
  $\partial(\text{anything})/\partial\phi$, the shape sensitivity. This is the entry point
  for inverse design: a relaxed obstacle is just a $\phi$ field with
  `requires_grad=True`.
- `torch.where(delta < 0.5, near, far)` evaluates both branches everywhere; that is fine
  and is the vectorised way to express the piecewise rule.
- The test uses a channel whose walls sit at $\delta = 0.2$ and $0.7$. The analytic
  Poiseuille profile then has width $H = (n_y - 3) + 0.2 + 0.7$, and the test checks the
  solver honours that width. Measured relative error: $6\times10^{-3}$ on 15 fluid rows.

Cylinder benchmark at $D = 20$, same grid, same everything except the wall rule:

| | Strouhal | $C_D$ | $C_{L,\max}$ |
|---|---|---|---|
| staircase halfway | 0.296 | 3.52 | 1.21 |
| Bouzidi | 0.302 | 3.25 | 0.94 |
| reference | 0.295-0.305 | 3.22-3.24 | 0.99-1.01 |

Drag error drops from 9% to under 1%, lift from 20% to 5%. Same cost. This is the
clearest demonstration in the repo of why boundary treatment, not collision, usually
limits LBM accuracy on curved geometry.

**Self-check.** For $\delta < 1/2$ the scheme reads a population from the *upstream* node
$\mathbf{x}_f - \mathbf{c}_q$. Why is that needed, and when could that node be solid?

## 8. Zou-He inlet and outlet (`boundary/zou_he.py`)

Bounce-back cannot impose a velocity profile or a pressure. Zou-He does: after streaming,
the three populations entering from outside the domain are unknown. Three conditions fix
them: the prescribed velocity (two components) and the assumption that the non-equilibrium
part of the normal population bounces back. Density at the inlet then follows from the
known populations, $\rho = [f_0 + f_2 + f_4 + 2(f_3 + f_6 + f_7)]/(1 - u_x)$. The outlet
is the mirror image with $\rho$ prescribed and $u_x$ solved for.

Two limits to know: the formulas are specific to D2Q9 and to the $x$-edges, and the
corner nodes are ambiguous (both the wall and the inlet want to set them). The convention
here is to apply bounce-back last so corners belong to the wall.

---

## 9. Force on a body by momentum exchange (`BounceBack.force`)

Every reflected population transfers momentum $\mathbf{c}_q(f_q^\star + f_{\bar q})$ to
the wall. Summing over fluid-to-solid links gives the total force with no surface
integration and no pressure reconstruction. It is exactly consistent with the bounce-back
rule, and it is differentiable, which is what inverse design needs.

Two bugs found while validating it, both instructive:

1. **Solid-to-solid links.** The periodic `roll` also flags links from the top solid row
   into the bottom solid row. Those nodes hold junk. The test (`tests/test_force.py`)
   compares wall force with the total body force in Poiseuille flow and was 56% off
   until `& ~solid` was added.
2. **Which solid?** The first cylinder benchmark had the cylinder and the channel walls
   in one `BounceBack`, so "drag on the cylinder" included friction on 440 lattice units
   of wall, giving $C_D = 5.4$ instead of about 3.2. Separate objects fixed it.

Lesson: a force test against a conservation identity (wall force = body force) is cheap
and catches a whole class of bookkeeping errors.

---

## 10. MRT collision (`collision/mrt.py`)

BGK has one rate for everything. MRT transforms to moment space with a fixed 9×9 matrix
$M$ (Lallemand-Luo ordering: $\rho, e, \varepsilon, j_x, q_x, j_y, q_y, p_{xx}, p_{xy}$),
relaxes each moment at its own rate, and transforms back. Only the two shear-stress rates
set the viscosity; the others are free and are chosen for stability.

Implementation: $C = M^{-1} S M$ is a 9×9 matrix applied along the population axis with
one `einsum`. Forcing uses the same trick with $(I - S/2)$. There is no loop over moments.

`tests/test_mrt.py::test_mrt_equals_bgk_for_uniform_rates` sets all rates to $1/\tau$ and
checks agreement with BGK to $10^{-13}$. That single test verifies $M$, $M^{-1}$, the
ordering, and the forcing algebra at once.

Why it is here: the cylinder benchmark at $Re = 100$ with $D = 20$ and $U = 0.05$ needs
$\tau = 0.53$, where BGK is at the edge of stability. MRT runs it comfortably. It is also
the natural host for a learned collision operator: a network predicting the free rates.

---

## 10b. TRT collision (`collision/trt.py`)

Between BGK (one rate) and MRT (nine) sits TRT (two). Split each population into a
symmetric part $f^+_q = (f_q + f_{\bar q})/2$ and an antisymmetric part
$f^-_q = (f_q - f_{\bar q})/2$; relax them at different rates. The symmetric rate sets
viscosity. The antisymmetric rate is free, and the product
$\Lambda = (\tau^+ - 1/2)(\tau^- - 1/2)$ controls where bounce-back puts the wall.

The payoff is precise: with $\Lambda = 3/16$ the wall is exactly midway between nodes at
*any* viscosity. Section 7 said BGK achieves this only at $\tau \approx 0.933$.
`tests/test_trt.py` runs Poiseuille at $\tau = 0.6$ and $\tau = 1.5$ and gets errors below
$10^{-4}$, where BGK gives about $10^{-2}$. The implementation is nine lines because the
opposite-direction index `lattice.opp` makes $f_{\bar q}$ a single fancy-index.

For most single-phase work TRT is the right default: BGK's cost, MRT's wall accuracy, and
one parameter to reason about.

**Self-check.** Why does setting $\Lambda = (\tau^+ - 1/2)^2$ recover BGK exactly?

## 11. Validation suite: what each test proves

| Test | What it establishes |
|---|---|
| `test_lattice` | velocity set is symmetric and isotropic |
| `test_macroscopic` | equilibrium reproduces its own moments |
| `test_poiseuille` | viscosity, forcing, and wall location are all right |
| `test_force` | momentum exchange is consistent with the body force |
| `test_interpolated` | Bouzidi walls honour off-node positions; differentiable in $\phi$ |
| `test_zou_he` | inlet/outlet reproduce uniform and pressure-driven flow |
| `test_mrt` | equal rates reduce to BGK; Poiseuille at $\tau = 0.6$ |
| `test_trt` | exact wall location at any $\tau$ with $\Lambda = 3/16$ |
| `test_taylor_green` | decay rate within 2%; **spatial order 2.008, 2.001** |
| `test_cavity` | Ghia et al. Re 100 centreline: 0.006 at 40k steps; the test asserts < 0.02 at 20k |
| `test_cylinder` (slow) | Bouzidi cylinder: St 0.302, $C_D$ 3.25, $C_{L,\max}$ 0.94 vs 0.30, 3.23, 1.00 |
| `test_autograd`, `test_checkpoint` | gradients match finite differences; checkpointing is exact |
| `test_smagorinsky` | reduces to BGK at $C_s = 0$ and in uniform flow; differentiable |

The figures for these cases are produced by `examples/cavity.py`,
`examples/learned_collision.py` and `examples/inverse_design.py`.

**Taylor-Green and convergence order.** The trick is *diffusive scaling*: keep $\tau$
fixed, scale the velocity amplitude as $1/N$ and the number of steps as $N^2$. Then the
physical end time and Reynolds number are fixed, the Mach number shrinks with $N$, and the
error ratio between grids gives the spatial order directly. Any other scaling mixes
compressibility error into the measurement.

**Self-check.** The Taylor-Green run initialises at equilibrium, ignoring the
non-equilibrium part. Why does the observed order still come out as 2?

---

## 12. Differentiability: how it works and what breaks it

Nothing special was done to "make the solver differentiable". It is differentiable
because every operation is a PyTorch tensor op and the rules below were followed:

- **No in-place writes** to tensors that may require grad. Build with `where`/`stack`.
- **No `.item()`, `.numpy()`, or Python floats** inside the loop. They cut the graph.
- **No NaNs in unselected `where` branches.** See section 7.
- **`sqrt` at zero** has infinite gradient. Smagorinsky adds `1e-30` inside the root.

`tests/test_autograd.py` uses `torch.autograd.gradcheck`, which compares the autograd
gradient with central finite differences in float64. It is the definitive test; run it
on a tiny grid for a few steps whenever you add a new operator.

**Gradient checkpointing** (`Simulation.run(..., checkpoint_every=k)`). Backpropagating
through $n$ steps normally stores $n$ copies of the populations. Wrapping each block of
$k$ steps in `torch.utils.checkpoint.checkpoint` stores only block boundaries and recomputes
the interior during backward. Memory falls from $O(n)$ to $O(n/k + k)$ for one extra
forward pass. `test_checkpoint` shows the gradient is identical to 12 digits. This is what
makes inverse problems over thousands of steps feasible on one GPU.

---

## 13. Smagorinsky LES (`collision/smagorinsky.py`)

The first turbulence model, chosen because it costs nothing extra in LBM: the
non-equilibrium stress $\Pi^{neq} = \sum_q \mathbf{c}_q\mathbf{c}_q (f_q - f_q^{eq})$ is
already proportional to the strain rate, so no finite differences are needed. The eddy
viscosity is folded into a *local* relaxation time via a closed-form quadratic (Hou et al.
1996).

Points to understand:

- $C_s$ is a tensor parameter. With `requires_grad=True` it is a one-parameter learnable
  closure, the simplest possible "solver-in-the-loop" model. A network predicting a
  $C_s$ field per node is the next step: a learned closure.
- The model is only meaningful at Reynolds numbers where the grid cannot resolve the
  flow. At $Re = 100$ it barely changes anything; that is expected.

---

## 14. Benchmarks module and the cylinder (`benchmarks.py`)

`schaefer_turek_cylinder` builds the 2D-2 benchmark: channel 22 by 4.1 diameters, cylinder
at (2, 2) diameters (slightly off-centre to trigger shedding), parabolic inlet with mean
speed $U$, outlet at $\rho = 1$. Reference values from Schäfer and Turek (1996):
Strouhal 0.295 to 0.305, $C_D$ 3.22 to 3.24, $C_{L,\max}$ 0.99 to 1.01.

Measured with $D = 20$ cells: with the default Bouzidi cylinder, Strouhal 0.302, $C_D$
3.25, $C_{L,\max}$ 0.94. With a staircase cylinder (`curved=False`) the forces are 9% and
20% high, the known signature of an unfitted wall at coarse resolution (table in
section 7b).

`examples/cylinder_animation.py` renders the wake as a GIF.

The first run reported $C_D = 5.4$. That was not a physics error: the cylinder and the
channel walls shared one `BounceBack`, so the "cylinder force" included wall friction.

The Strouhal number is extracted from upward zero crossings of the lift signal. That is
robust to a slowly drifting mean and needs no FFT.

---

## 15. Machine learning inside the solver

Everything before this point was classical CFD engineered so that autograd works. This
part uses it. Three problems, in increasing order of ambition, each with a test and an
example script. Read the scripts in `examples/`; they are short.

### 15.1 Parameter inversion (`inverse.py`, `examples/inverse_viscosity.py`)

**Problem.** A Taylor-Green vortex decays with an unknown viscosity. We observe the
velocity at 10% of the nodes, once, after 60 steps, with 0.5% noise. Find $\tau$.

**Method.** Define the misfit $J(\tau) = \sum_{\text{samples}} |u_\tau - u_{\text{obs}}|^2$
where $u_\tau$ comes from running the solver. Autograd gives $dJ/d\tau$ by
backpropagating through all 60 steps (with checkpointing every 20). Hand $J$ and its
gradient to L-BFGS.

**Two details that matter.**

- *Parameterisation.* The solver is unstable for $\tau < 1/2$. `PositiveTau` stores an
  unconstrained $\theta$ and exposes $\tau = 1/2 + \mathrm{softplus}(\theta)$, so no
  optimiser step can leave the stable region. This is the standard trick for any
  bounded physical parameter.
- *Optimiser choice.* Adam with a fixed step oscillated around the answer (the test
  first showed $\tau$ bouncing between 0.77 and 0.88). For a handful of parameters with
  a cheap, smooth objective, L-BFGS with a strong-Wolfe line search converges in one
  call to $|\tau - \tau_{\text{true}}| < 10^{-3}$. Adam is for thousands of parameters
  and noisy gradients, not for this.

The example plots the full misfit landscape next to the L-BFGS iterates, so you can see
that the problem is well posed (one clean minimum) and how few evaluations it took.

**Self-check.** What happens to the landscape if you observe only the density instead of
the velocity? Think about how $\rho$ depends on $\nu$ in Taylor-Green.

### 15.2 Shape inversion and inverse design (`geometry.py`, `examples/inverse_design.py`)

**Level sets as design variables.** Section 7b made the wall position a function of a
tensor $\phi$. `geometry.py` builds $\phi$ for circles and ellipses from a few scalar
parameters. Gradients then flow: loss $\to$ velocity field $\to$ Bouzidi reflection
$\to$ $\delta$ per link $\to$ $\phi$ $\to$ radius or aspect ratio.

**Shape inversion** (`tests/test_inverse_shape.py`): recover the radius of a cylinder
from velocity samples downstream. L-BFGS starting from 3.0 lands within 0.1 of the true
4.0. Note what the gradient is *not*: the set of solid nodes changes discretely as the
radius crosses a node, and that jump has no gradient. Only $\delta$ moves continuously.
The optimiser still works because the continuous part dominates on a fine enough grid.

**Inverse design.** Fix the area of an ellipse, free its aspect ratio, minimise drag
from the momentum-exchange force. The result is what intuition predicts: the body
elongates along the flow. The test checks only the sign of the gradient at the circle
(it must be negative); the example runs the optimisation and plots start, end, and the
drag curve.

Nothing here needed an adjoint solver to be written: the adjoint *is* autograd.

**Self-check.** Why must the area be constrained? What would the optimiser do without it?

### 15.3 Learned collision, solver-in-the-loop (`learned.py`, `examples/learned_collision.py`)

**The idea.** A collision operator is a local map from populations to populations. Any
such map that preserves mass and momentum and is stable is a candidate. `LearnedMRT`
keeps standard MRT and adds a correction $M^{-1}\,\delta\mathbf{m}$ from a small MLP,
where $\delta\mathbf{m}$ touches only the six non-conserved moments. Three design choices
carry the whole thing:

1. **Conservation by construction.** Rows of $\delta\mathbf{m}$ for $\rho$, $j_x$, $j_y$
   are identically zero. `tests/test_learned.py` fills the network with random weights
   and checks mass and momentum are still conserved to machine precision. A network that
   can violate conservation would need a penalty term and would still leak; this one
   cannot leak.
2. **Zero-initialised last layer.** An untrained model is *exactly* MRT. Training can
   only move away from a known-good solver, never start from garbage.
3. **Local invariant features.** The network sees $(\rho - 1, u_x, u_y, \Pi^{neq}_{xx},
   \Pi^{neq}_{xy}, \Pi^{neq}_{yy})$ at one node, scaled to O(1). No neighbours, no
   coordinates. It can therefore be applied on any grid size.

**Training.** Unroll the solver 30 steps on a coarse $16^2$ Taylor-Green at Mach 0.14,
where compressibility error is a few percent, and compare with the analytic solution at
steps 10, 20, 30. Backpropagate through the trajectory (this is what "solver-in-the-loop"
means). Adam, 150 iterations, three seconds on a laptop CPU.

**Result.** Relative velocity error² drops from $5.7\times10^{-4}$ to $8.9\times10^{-5}$
on the training case, and from $5.8\times10^{-4}$ to $9.3\times10^{-5}$ on a held-out
case with a different amplitude and a shifted phase: a 6x reduction that generalises
across the flow family.

**The catch, measured.** Evaluate the same model on a longer horizon than it was
trained on and it over-corrects:

| trained on | evaluated at 30 steps | 60 steps | 120 steps |
|---|---|---|---|
| plain MRT | 2.40e-2 | 2.35e-2 | 2.28e-2 |
| learned, 30-step horizon | 3.66e-3 | 2.46e-2 | 4.68e-2 |
| learned, 60-step horizon | 1.23e-2 | 8.09e-3 | 2.72e-2 |

(relative velocity error at the final step, held-out case). Look at the right panel of
`learned_collision.png` (from `examples/learned_collision.py`): plain MRT settles at a constant 2.3% error, which is the
*initial-layer* artefact of starting from equilibrium populations (section 11). The
network learned to cancel that artefact by step 30, then keeps pushing. It learned "push
the error down by step 30", not "be a better collision operator". Training on 60 steps moves
the useful window to 60 and is worse at 30. This is the central practical lesson of
solver-in-the-loop training and the reason the literature unrolls as far as memory
allows (Um et al. 2020, Kochkov et al. 2021): **match the training horizon to the
deployment horizon**, or randomise it. Gradient checkpointing (section 12) is what makes
long horizons affordable. The example script prints the 2x-longer case so you see this
yourself.

**Honest limits.** The reference is an incompressible analytic solution, so the network
is learning to cancel compressibility and discretisation error of this particular
family. It is a demonstration of the mechanism, not a general closure. The next step
toward a general learned closure is to train against filtered high-resolution data on
an under-resolved turbulent flow, with Smagorinsky (section 13) as the baseline to beat.

**Self-check.** The correction is added *after* MRT rather than replacing it. What breaks
if the network instead outputs the full post-collision state?

## 16. Performance (`examples/benchmark_mlups.py`)

See the table in this section once the benchmark has run on your machine. The
questions to ask of any such table:

- Does throughput per node rise with grid size? If yes, the small grids are
  launch-bound and the large ones are bandwidth-bound.
- Does `torch.compile` help? It fuses elementwise ops (collision) but cannot fuse the
  rolls into a single stencil pass, so expect a moderate gain, not 10x.
- What is the roofline? D2Q9 reads and writes 9 doubles per node per step: 144 bytes.
  At 100 GB/s memory bandwidth that caps at about 700 MLUPS. Anything far below that
  is overhead, not physics.

Triton is the tool for going further: one kernel that reads 9 neighbours, collides, and
writes 9 outputs, so every byte moves once. It is CUDA-only, so it is deferred until a
CUDA machine is available; the pure-torch path stays as the reference implementation.

## 17. Performance notes (measured)

`examples/benchmark_mlups.py`, BGK plus one moving-wall bounce-back, Apple M-series laptop,
measured while other jobs were running so treat as ±20%:

| Grid | Device | dtype | eager MLUPS | `torch.compile` MLUPS |
|---|---|---|---|---|
| 64² | CPU | f32 | 5.5 | 6.6 |
| 64² | MPS | f32 | 0.5 | 0.5 |
| 128² | CPU | f32 | 15.9 | 16.2 |
| 128² | MPS | f32 | 1.8 | 1.8 |
| 256² | CPU | f32 | 12.2 | 15.9 |
| 256² | MPS | f32 | 3.9 | 4.1 |
| 512² | CPU | f32 | 13.9 | 16.2 |
| 512² | CPU | f64 | 5.1 | 7.6 |
| 512² | MPS | f32 | 6.1 | 8.2 |

What the table says:

- **MPS throughput grows 12x from 64² to 512²** while CPU is flat. That is the signature
  of per-kernel launch overhead: at 64² each of the ~60 kernels per step does 4k
  elements of work and the launch cost dominates. The GPU only starts to pay off above
  about 512², and would keep rising on larger grids.
- **`torch.compile` gives 10 to 50%**, more in float64. It fuses the elementwise
  collision chain but cannot fuse the nine `roll`s with it, so the stencil still costs
  separate passes.
- **Roofline.** 144 bytes per node per step in float64 at roughly 100 GB/s gives a
  ceiling near 700 MLUPS. We are at 1 to 2% of it. Closing the gap needs a fused
  kernel that reads the nine neighbours once, collides, and writes once. That is a
  Triton (CUDA) job and needs a CUDA machine.

The older cylinder numbers (2.5 to 4 MLUPS on a 440×84 grid with MRT and four boundary
objects) fit the same picture: more kernels per step (nine rolls, nine `where`s per
bounce-back object, einsums), each doing too little work on a 37k-node grid to amortise
its launch. Future work: a Triton kernel that fuses collide, stream and bounce-back.

---

## 18. Reading list, in order

1. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice* (2017).
   Chapters 3, 4, 5, 6, 10, 14. The one book to own.
2. Guo, Zheng, Shi, *Phys. Rev. E* 65, 046308 (2002). Forcing.
3. Lallemand and Luo, *Phys. Rev. E* 61, 6546 (2000). MRT.
4. Zou and He, *Phys. Fluids* 9, 1591 (1997). Inlet/outlet.
5. Ladd, *J. Fluid Mech.* 271, 285 (1994). Moving walls and momentum exchange.
5a. Bouzidi, Firdaouss, Lallemand, *Phys. Fluids* 13, 3452 (2001). Curved walls.
5b. Ginzburg, Verhaeghe, d'Humières, *Commun. Comput. Phys.* 3, 427 (2008). TRT and the
   magic parameter.
6. Hou et al., *Fields Inst. Comm.* 6, 151 (1996). Smagorinsky in LBM.
7. Schäfer and Turek (1996). Cylinder benchmark definitions and reference values.
8. Bezgin, Buhendwa, Adams, *Comput. Phys. Commun.* 282 (2023): JAX-Fluids, for how
   differentiable CFD codes are structured. Same ideas, finite volume.
9. Um, Brand, Fei, Holl, Thuerey, *NeurIPS* (2020), "Solver-in-the-loop". The paper that
   named the training strategy in section 15.3.
10. Kochkov et al., *PNAS* 118 (2021), "Machine learning accelerated CFD". Learned
    corrections inside a classical solver, with the evaluation methodology worth copying.
11. Nocedal and Wright, *Numerical Optimization*, chapter 7, for why L-BFGS wins in
    section 15.1.

---

## 19. Glossary

- **Population** $f_q$: distribution function in direction $q$.
- **Lattice units**: $\Delta x = \Delta t = 1$; convert physical quantities before running.
- **$\tau$**: relaxation time; $\nu = c_s^2(\tau - 1/2)$.
- **MLUPS**: million lattice-site updates per second, the standard LBM throughput unit.
- **Halfway bounce-back**: no-slip wall located midway between fluid and solid nodes.
- **Momentum exchange**: force on a body from the populations it reflects.
- **Diffusive scaling**: refinement with $U \propto 1/N$ and steps $\propto N^2$.
- **Gradient checkpointing**: recompute instead of store, for memory-bounded backprop.
- **Solver-in-the-loop**: training a model by unrolling the numerical solver it sits in
  and backpropagating through the trajectory.
- **Level set**: a scalar field whose zero contour is the wall; negative inside solid.
- **Momentum exchange**: force on a body from the populations it reflects.
