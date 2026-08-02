"""
Capstone Project 6 (Case Study 6): Cook's membrane -- nearly
incompressible elasticity. Compares a pure-displacement P1 formulation
(which locks as nu -> 0.5), a pure-displacement P2 formulation (locks
less severely but still measurably), and a genuine mixed
displacement-pressure P2/P1 formulation (locking-free across the full
range of Poisson ratio) using real scikit-fem assembly.

Geometry (the classical Cook's membrane benchmark): a tapered
cantilever panel with vertices (0,0), (48,44), (48,60), (0,44) [mm],
clamped at x=0, subjected to a unit in-plane shear traction on the
x=48 edge.

Reported quantity: vertical tip displacement at (48, 52) (edge
midpoint), tracked as a function of Poisson ratio nu in
{0.3, 0.45, 0.49, 0.499, 0.4999} for all three formulations.

Reference:
  R.D. Cook, "Improved two-dimensional finite element," ASCE J.
  Struct. Div. 100(9), 1974, pp. 1851-1863 (origin of the benchmark
  geometry and loading).
  D. Boffi, F. Brezzi, M. Fortin, "Mixed Finite Element Methods and
  Applications," Springer, 2013, Ch. 8 (nearly incompressible
  elasticity, volumetric locking).
"""
import numpy as np
import gmsh
from skfem import *
from skfem.io.meshio import from_file
from skfem.helpers import dot, ddot, sym_grad, div, grad
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve


def build_cooks_mesh(filename="cooks.msh", lc=3.0):
    gmsh.initialize()
    gmsh.model.add("cooks")
    pts = [(0, 0), (48, 44), (48, 60), (0, 44)]
    tags = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in pts]
    lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [lines[3]], name="clamped")   # x=0 edge
    gmsh.model.addPhysicalGroup(1, [lines[1]], name="loaded")    # x=48 edge
    gmsh.model.addPhysicalGroup(2, [surf], name="body")
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()


build_cooks_mesh("cooks.msh", lc=3.0)
mesh = from_file("cooks.msh", out=[])

E = 1.0
traction = np.array([0.0, 1.0 / 16.0])  # unit total shear force distributed on loaded edge (length 16)


def solve_displacement_only(nu, degree=1):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    Ev = ElementVector(ElementTriP1()) if degree == 1 else ElementVector(ElementTriP2())
    bv = Basis(mesh, Ev, intorder=4)

    @BilinearForm
    def a(u, v, w):
        return 2 * mu * ddot(sym_grad(u), sym_grad(v)) + lam * div(u) * div(v)

    A = a.assemble(bv)

    fb = FacetBasis(mesh, Ev, facets=mesh.boundaries["loaded"], intorder=4)

    @LinearForm
    def load(v, w):
        return dot(traction, v)

    F = load.assemble(fb)
    D = bv.get_dofs(mesh.boundaries["clamped"])
    Acon, Fcon, x0, I = condense(A, F, D=D)
    x = solve(Acon, Fcon)
    u_full = np.zeros(A.shape[0])
    u_full[I] = x

    tip_dofs = bv.get_dofs(mesh.boundaries["loaded"])
    probe_pt = np.array([[48.0], [52.0]])
    ux = bv.probes(probe_pt) @ u_full
    return ux


def solve_mixed(nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    Ev = ElementVector(ElementTriP2())
    Ep = ElementTriP1()
    bv = Basis(mesh, Ev, intorder=4)
    bp = Basis(mesh, Ep, intorder=4)

    @BilinearForm
    def a_uu(u, v, w):
        return 2 * mu * ddot(sym_grad(u), sym_grad(v))

    @BilinearForm
    def b_up(u, p, w):
        return -div(u) * p

    @BilinearForm
    def c_pp(p, q, w):
        return -(1.0 / lam) * p * q

    A = a_uu.assemble(bv)
    B = b_up.assemble(bv, bp)
    C = c_pp.assemble(bp)

    K = bmat([[A, B.T], [B, C]], "csc")

    fb = FacetBasis(mesh, Ev, facets=mesh.boundaries["loaded"], intorder=4)

    @LinearForm
    def load(v, w):
        return dot(traction, v)

    F = load.assemble(fb)
    rhs = np.concatenate([F, np.zeros(bp.N)])

    Dv = bv.get_dofs(mesh.boundaries["clamped"])
    fixed = Dv.flatten()
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    x = np.zeros(K.shape[0])
    x[free] = spsolve(K[free][:, free], rhs[free])

    u_full = x[:bv.N]
    probe_pt = np.array([[48.0], [52.0]])
    ux = bv.probes(probe_pt) @ u_full
    return ux[1]


print(f"{'nu':>10} {'lambda':>12} {'tip v (P1)':>12} {'tip v (P2)':>12} "
      f"{'tip v (mixed P2/P1)':>20}")
for nu in [0.3, 0.45, 0.49, 0.499, 0.4999]:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    v_p1 = solve_displacement_only(nu, degree=1)[1]
    v_p2 = solve_displacement_only(nu, degree=2)[1]
    v_mixed = solve_mixed(nu)
    print(f"{nu:10.4f} {lam:12.2f} {v_p1:12.5f} {v_p2:12.5f} {v_mixed:20.5f}")
