# Lattice Boltzmann method: notes

Lattice units throughout: Δx = Δt = 1.

## Discrete kinetic equation

$$f_q(\mathbf{x} + \mathbf{c}_q, t + 1) = f_q(\mathbf{x}, t) + \Omega_q + \left(1 - \tfrac{1}{2\tau}\right) S_q$$

Macroscopic fields are moments of the populations:

$$\rho = \sum_q f_q, \qquad \rho\mathbf{u} = \sum_q \mathbf{c}_q f_q + \tfrac{1}{2}\mathbf{F}.$$

## BGK collision

$\Omega_q = -\frac{1}{\tau}(f_q - f_q^{eq})$ with the second-order equilibrium

$$f_q^{eq} = w_q \rho \left[ 1 + \frac{\mathbf{c}_q\cdot\mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_q\cdot\mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u}\cdot\mathbf{u}}{2c_s^2} \right].$$

Chapman-Enskog recovers weakly compressible Navier-Stokes with $p = c_s^2\rho$ and

$$\nu = c_s^2\left(\tau - \tfrac{1}{2}\right), \qquad c_s^2 = \tfrac{1}{3}.$$

## Forcing (Guo et al. 2002)

$$S_q = w_q \left[ \frac{\mathbf{c}_q - \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_q\cdot\mathbf{u})\,\mathbf{c}_q}{c_s^4} \right]\cdot\mathbf{F}$$

together with the half-force shift in the velocity definition above.

## MRT collision (Lallemand and Luo 2000)

Collision in moment space, $\mathbf{m} = M\mathbf{f}$, with moments ordered
$(\rho, e, \varepsilon, j_x, q_x, j_y, q_y, p_{xx}, p_{xy})$:

$$\mathbf{f}^\star = \mathbf{f} - M^{-1} S (\mathbf{m} - \mathbf{m}^{eq}) + M^{-1}\left(I - \tfrac{1}{2}S\right) M\,\mathbf{S}^{\mathrm{Guo}}$$

$S = \mathrm{diag}(s_q)$. The shear rates $s_7 = s_8 = 1/\tau$ set $\nu$; the others are free and default to $(s_e, s_\varepsilon, s_q) = (1.64, 1.54, 1.2)$. Equilibrium moments are $M\mathbf{f}^{eq}$ with the BGK equilibrium. Setting all rates to $1/\tau$ recovers BGK exactly (tested). MRT stays stable to lower $\tau$ than BGK, which is what the cylinder benchmark at $\tau = 0.53$ needs.

## TRT collision (Ginzburg et al. 2008)

Split populations into symmetric and antisymmetric parts, $f_q^\pm = (f_q \pm f_{\bar q})/2$, relaxed at $\omega^\pm = 1/\tau^\pm$. $\tau^+$ sets $\nu$; $\tau^-$ follows from the magic parameter $\Lambda = (\tau^+ - \tfrac12)(\tau^- - \tfrac12)$. With $\Lambda = 3/16$ halfway bounce-back is exactly midway at any viscosity; tested: Poiseuille error $< 10^{-4}$ at $\tau = 0.6$ and $1.5$, where BGK gives $10^{-2}$. $\Lambda = 1/4$ is the most stable choice.

## Smagorinsky LES (Hou et al. 1996)

The non-equilibrium stress $\Pi^{neq}_{\alpha\beta} = \sum_q c_{q\alpha}c_{q\beta}(f_q - f_q^{eq})$ is proportional to the resolved strain rate, so the eddy viscosity $\nu_t = (C_s\Delta)^2|S|$ folds into a local relaxation time in closed form:

$$\tau_{\mathrm{eff}} = \tfrac{1}{2}\left[\tau_0 + \sqrt{\tau_0^2 + 18\sqrt{2}\, C_s^2 |\Pi^{neq}|/\rho}\right], \qquad |\Pi^{neq}| = \sqrt{\Pi^{neq}_{\alpha\beta}\Pi^{neq}_{\alpha\beta}}.$$

$C_s = 0$ recovers BGK. $C_s$ may be a learnable tensor.

## Halfway bounce-back

A post-collision population at a fluid node that would stream into a solid node is reflected back to the same node with reversed direction. The effective wall lies midway between the last fluid node and the first solid node. For a moving wall with velocity $\mathbf{u}_w$ (Ladd 1994):

$$f_{\bar q}(\mathbf{x}, t+1) = f_q^\star(\mathbf{x}, t) - 2 w_q \rho \frac{\mathbf{c}_q\cdot\mathbf{u}_w}{c_s^2}.$$

The hydrodynamic force on the solid follows by momentum exchange over all fluid-to-solid links:

$$\mathbf{F} = \sum_{\mathbf{x}\in\mathrm{fluid}}\ \sum_{q:\ \mathbf{x}+\mathbf{c}_q\in\mathrm{solid}} \mathbf{c}_q\left[f_q^\star(\mathbf{x}) + f_{\bar q}(\mathbf{x}, t+1)\right].$$

Solid-to-solid links must be excluded: solid nodes hold junk populations.

With BGK the wall is exactly midway only when $(\tau - 1/2)^2 = 3/16$, i.e. $\tau \approx 0.933$; other values give a small viscosity-dependent slip.

## Interpolated bounce-back (Bouzidi et al. 2001)

Walls are described by a level set $\phi(\mathbf{x})$, negative in the solid. For a fluid node $\mathbf{x}_f$ whose neighbour $\mathbf{x}_s = \mathbf{x}_f + \mathbf{c}_q$ is solid, the fluid fraction of the link is $\delta = \phi(\mathbf{x}_f)/(\phi(\mathbf{x}_f) - \phi(\mathbf{x}_s))$ and

$$f_{\bar q}(\mathbf{x}_f, t+1) = \begin{cases} 2\delta f_q^\star(\mathbf{x}_f) + (1-2\delta) f_q^\star(\mathbf{x}_f - \mathbf{c}_q), & \delta < \tfrac12 \\ \dfrac{1}{2\delta} f_q^\star(\mathbf{x}_f) + \dfrac{2\delta-1}{2\delta} f_{\bar q}^\star(\mathbf{x}_f), & \delta \ge \tfrac12. \end{cases}$$

$\delta = \tfrac12$ is halfway bounce-back. Second order for curved walls, and differentiable in $\phi$, so the wall position is an optimisation variable. Momentum exchange uses the same reflected populations.

## Zou--He inlet and outlet

For D2Q9 flow in the $x$ direction, the left inlet prescribes $(u_x,u_y)$ and
recovers density from populations known after streaming:

$$\rho_{\mathrm{in}}=\frac{f_0+f_2+f_4+2(f_3+f_6+f_7)}{1-u_x}.$$

The right outlet prescribes density (equivalently pressure $p=c_s^2\rho$)
and transverse velocity, then recovers the normal velocity:

$$u_x=\frac{f_0+f_2+f_4+2(f_1+f_5+f_8)}{\rho_{\mathrm{out}}}-1.$$

Unknown incoming populations are reconstructed by bounce-back of the
non-equilibrium distribution. These conditions act after streaming. Apply wall
bounce-back after the inlet and outlet so solid corner nodes remain governed by
the wall condition. The current boundary interface does not pass body force,
so use these conditions for unforced flow.

## Limits

- Keep $|\mathbf{u}|/c_s \lesssim 0.1$ (velocities of a few hundredths in lattice units) to limit compressibility error.
- BGK is unstable as $\tau \to 1/2$; practical floor around 0.52. MRT or TRT collision is the standard remedy.

## Differentiability and memory

Every operation in the loop is a tensor op, so `torch.autograd` differentiates the whole run with respect to $\tau$, $\mathbf{F}$, initial state, inlet velocity, outlet density, $C_s$, and MRT rates. Backpropagating through $n$ steps stores $n$ population fields. `Simulation.run(n, checkpoint_every=k)` wraps blocks of $k$ steps in `torch.utils.checkpoint`, storing only block boundaries and recomputing interiors on the backward pass: memory $O(n/k + k)$ for one extra forward.

## Inverse problems and learned components

- **Parameter inversion.** Minimise $J(\tau) = \sum_{\text{obs}} |u_\tau - u_{\text{obs}}|^2$ with $dJ/d\tau$ from autograd through the unrolled solver. $\tau = \tfrac12 + \mathrm{softplus}(\theta)$ keeps the search in the stable region. L-BFGS with strong-Wolfe line search.
- **Shape.** Level set $\phi(\mathbf{x}; p)$ with design parameters $p$ (radius, semi-axes). Sensitivities flow through Bouzidi's $\delta = \phi_f/(\phi_f - \phi_s)$. The solid-node set changes discretely with $p$; only $\delta$ is smooth.
- **Learned collision.** $\mathbf{f}^\star = \mathrm{MRT}(\mathbf{f}) + M^{-1}\delta\mathbf{m}(\mathbf{z})$ with $\delta m$ zero on $(\rho, j_x, j_y)$, so conservation is exact for any network. Features $\mathbf{z} = (\rho-1, u_x, u_y, \Pi^{neq}_{xx}, \Pi^{neq}_{xy}, \Pi^{neq}_{yy})$. Trained by unrolling 30 steps against the analytic Taylor-Green solution.

## Validation

| Test | Reference | Result |
|---|---|---|
| `test_poiseuille.py` | $u_x = \frac{g}{2\nu}y(H-y)$, $H$ = fluid rows | rel. error $< 5\times10^{-3}$ at $\tau = 0.933$ |
| `test_force.py` | wall force = total body force | rel. error $< 10^{-3}$ |
| `test_interpolated.py` | $\delta = 1/2$ equals halfway; Poiseuille with walls at $\delta = 0.2, 0.7$; grad w.r.t. $\phi$ | exact; $< 10^{-2}$ (measured $6\times10^{-3}$); finite |
| `test_inverse_viscosity.py` | recover $\tau = 0.8$ from 10% velocity samples | $|\Delta\tau| < 10^{-3}$ in one L-BFGS call |
| `test_inverse_shape.py` | recover cylinder radius 4.0 from samples; drag gradient sign for streamlining | $|\Delta r| < 0.1$; $dF_x/d\log(a/b) < 0$ |
| `test_learned.py` | untrained = MRT; conservation with random weights; error after training | exact; exact; train and held-out error $< 1/3$ of MRT |
| `test_zou_he.py` | uniform flow exact; pressure-driven channel | $< 3\%$ |
| `test_smagorinsky.py` | $C_s = 0$ and uniform flow equal BGK; $\tau_{\rm eff} \ge \tau_0$ under shear | exact; differentiable in $C_s$ |
| `test_checkpoint.py` | checkpointed gradient equals plain gradient | rel. diff $< 10^{-12}$ |
| `test_trt.py` | TRT with $\tau^- = \tau^+$ equals BGK; Poiseuille at $\tau = 0.6, 1.5$ | $10^{-13}$; $< 10^{-4}$ |
| `test_mrt.py` | MRT with equal rates = BGK; Poiseuille at $\tau = 0.6$ | $10^{-13}$; $< 3\%$ |
| `test_taylor_green.py` | $\mathbf{u} \propto e^{-2\nu k^2 t}$; diffusive-scaling refinement 16/32/64 | decay rate within 2%; orders 2.008, 2.001 |
| `test_cavity.py` | Ghia et al. (1982) Re 100 centreline, 64² grid | max deviation 0.006 (40k steps), $< 0.02$ at 20k |
| `test_cylinder.py` (slow) | Schäfer and Turek (1996) 2D-2, Re 100, $D = 20$ | Bouzidi: St 0.302, $C_D$ 3.25, $C_{L,\max}$ 0.94. Staircase: 0.296, 3.52, 1.21. Ref: 0.295-0.305, 3.22-3.24, 0.99-1.01 |

Taylor-Green initial condition: $\mathbf{u} = U(-\cos kx \sin ky,\ \sin kx \cos ky)$, $\rho = 1 - \frac{U^2}{4c_s^2}(\cos 2kx + \cos 2ky)$, initialised at equilibrium. Under diffusive scaling ($\tau$ fixed, $U \propto 1/N$, steps $\propto N^2$) the missing non-equilibrium part is $O(N^{-2})$, so the observed order is clean.

## References

- T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer, 2017.
- M. Bouzidi, M. Firdaouss, P. Lallemand, Phys. Fluids 13, 3452 (2001).
- Z. Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 046308 (2002).
- I. Ginzburg, F. Verhaeghe, D. d'Humières, Commun. Comput. Phys. 3, 427 (2008).
- S. Hou, J. Sterling, S. Chen, G. D. Doolen, Fields Inst. Comm. 6, 151 (1996).
- P. Lallemand, L.-S. Luo, Phys. Rev. E 61, 6546 (2000).
- A. J. C. Ladd, J. Fluid Mech. 271, 285 (1994).
- U. Ghia, K. N. Ghia, C. T. Shin, J. Comput. Phys. 48, 387 (1982).
- M. Schäfer, S. Turek, in *Flow Simulation with High-Performance Computers II*, Vieweg (1996).
- Q. Zou and X. He, *Physics of Fluids* 9, 1591--1598 (1997).
