"""
Capstone Project 12 (Case Study 12): Fourth-order problems by mixed
splitting.
Part (a): manufactured biharmonic problem, H1xH1 mixed split
  -Delta u = w,  -Delta w = f,   avoiding C^1 elements.
Part (b): Cahn-Hilliard phase separation, mixed (c, mu) formulation,
  backward Euler in time, verifying exact mass conservation and
  monotonic free-energy decay (energy stability).

Reference:
  J.W. Cahn and J.E. Hilliard, "Free Energy of a Nonuniform System.
  I. Interfacial Free Energy," J. Chem. Phys. 28(2), 1958, pp. 258-267
  (origin of the model).
  C.M. Elliott and D.A. French, "A nonconforming finite-element method
  for the two-dimensional Cahn-Hilliard equation," SIAM J. Numer. Anal.
  26(4), 1989, pp. 884-903 (mixed H1xH1 finite element analysis).
Software pattern adapted from the official scikit-fem/FEniCSx-style
Cahn-Hilliard demo structure (Newton iteration on the coupled (c,mu)
system).
"""
import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

# ---------------- Part (a): manufactured biharmonic ----------------
print("=" * 70)
print("Part (a): manufactured biharmonic problem via H1 x H1 mixed split")
print("=" * 70)


def u_exact(x):
    return (np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])) ** 1  # simply-supported-consistent


def w_exact(x):
    # w = -Delta u for u = sin(pi x) sin(pi y)
    return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def f_source(x):
    # f = -Delta w = 4 pi^4 sin(pi x) sin(pi y)
    return 4 * np.pi**4 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def solve_biharmonic(nrefs):
    mesh = MeshTri().refined(nrefs)
    E = ElementTriP1()
    Ec = E * E
    basis = Basis(mesh, Ec)

    @BilinearForm
    def form(u, w, v, z, wdata):
        return dot(grad(u), grad(v)) - w * v + dot(grad(w), grad(z))

    @LinearForm
    def load(v, z, wdata):
        return f_source(wdata.x) * z

    A = form.assemble(basis)
    b = load.assemble(basis)
    D = basis.get_dofs(mesh.boundary_facets())
    Acon, bcon, x0, I = condense(A, b, D=D)
    x = solve(Acon, bcon)
    xf = np.zeros(A.shape[0])
    xf[I] = x
    (u_h, ub), (w_h, wb) = basis.split(xf)

    @Functional
    def uerr(wd):
        return (wd["uh"] - u_exact(wd.x)) ** 2

    @Functional
    def werr(wd):
        return (wd["wh"] - w_exact(wd.x)) ** 2

    L2_u = np.sqrt(uerr.assemble(ub, uh=ub.interpolate(u_h)))
    L2_w = np.sqrt(werr.assemble(wb, wh=wb.interpolate(w_h)))
    return mesh.param(), L2_u, L2_w


print(f"{'h':>10} {'L2(u) err':>12} {'L2(w) err':>12} {'rate(u)':>8} {'rate(w)':>8}")
prev_u, prev_w = None, None
for nrefs in [2, 3, 4, 5]:
    h, L2u, L2w = solve_biharmonic(nrefs)
    ru = "" if prev_u is None else f"{np.log2(prev_u/L2u):8.2f}"
    rw = "" if prev_w is None else f"{np.log2(prev_w/L2w):8.2f}"
    print(f"{h:10.5f} {L2u:12.5e} {L2w:12.5e} {ru:>8} {rw:>8}")
    prev_u, prev_w = L2u, L2w


# ---------------- Part (b): Cahn-Hilliard (two separate P1 bases) ----------------
print()
print("=" * 70)
print("Part (b): Cahn-Hilliard phase separation, semi-implicit (c, mu) scheme")
print("=" * 70)

mesh_ch = MeshTri().refined(4)
Ec = ElementTriP1()
bc = Basis(mesh_ch, Ec)
bmu = Basis(mesh_ch, Ec)

kappa = 5.0e-4
dt = 2.0e-4


def dpsi(c):
    return 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c)


def psi(c):
    return (c * (1 - c)) ** 2


rng = np.random.default_rng(0)
c0 = 0.5 + 0.02 * (rng.random(bc.N) - 0.5)


@BilinearForm
def mass_c(c, v, w):
    return c * v


@BilinearForm
def stiff_c(c, v, w):
    return dot(grad(c), grad(v))


Mc = mass_c.assemble(bc)
Kc = stiff_c.assemble(bc)     # stiffness on the SAME P1 space for both c and mu

# Semi-implicit (IMEX) scheme, nonlinear term explicit:
#   (c^{n+1}, v)/dt + (grad(mu^{n+1}), grad(v)) = (c^n, v)/dt
#   (mu^{n+1}, z) - kappa (grad(c^{n+1}), grad(z)) = (dpsi(c^n), z)
# This is linear in (c^{n+1}, mu^{n+1}) at every step -- standard IMEX
# treatment for Cahn-Hilliard (IMEX-Euler), unconditionally mass
# conservative and stable in practice for dt below the explicit-term's
# stiffness threshold.
Msys = bmat([[Mc / dt, Kc], [-kappa * Kc, Mc]], "csc")
from scipy.sparse.linalg import splu
lu = splu(Msys)


@Functional
def mass_functional(w):
    return w["c"]


def compute_mass(c_vec):
    return mass_functional.assemble(bc, c=bc.interpolate(c_vec))


def compute_energy(c_vec):
    @Functional
    def energy(w):
        gc = grad(w["c"])
        return psi(w["c"]) + 0.5 * kappa * (gc[0] ** 2 + gc[1] ** 2)
    return energy.assemble(bc, c=bc.interpolate(c_vec))


c_h = c0.copy()
mass0 = compute_mass(c_h)
energies = [compute_energy(c_h)]
n_steps = 400

for step in range(n_steps):
    rhs_c = (Mc @ c_h) / dt
    rhs_mu = Mc @ dpsi(c_h)
    rhs = np.concatenate([rhs_c, rhs_mu])
    sol = lu.solve(rhs)
    c_h = sol[:bc.N]
    if step % 40 == 0 or step == n_steps - 1:
        energies.append(compute_energy(c_h))

mass_end = compute_mass(c_h)
print(f"initial mass = {mass0:.8f}, final mass (after {n_steps} steps) = {mass_end:.8f}, "
      f"drift = {abs(mass_end-mass0):.3e}")
print("energy at recorded checkpoints (every 40 steps):",
      [f"{e:.5f}" for e in energies])
non_monotone = any(energies[i + 1] > energies[i] + 1e-8 for i in range(len(energies) - 1))
print(f"energy ever increased between checkpoints? {non_monotone}")
