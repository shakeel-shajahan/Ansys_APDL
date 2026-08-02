"""
Capstone Project 5 (Case Study 5): DFG 2D-1 benchmark (Schafer & Turek,
1996) -- steady laminar flow around a cylinder, Re=20, solved with
Taylor-Hood (P2/P1) elements on a real gmsh-generated curved-boundary
mesh, using Picard iteration for the convective nonlinearity, on THREE
mesh resolutions to demonstrate mesh convergence toward the published
benchmark interval.

Geometry (exact benchmark values): channel [0,2.2] x [0,0.41], cylinder
of diameter D=0.1 centered at (0.2, 0.2). Parabolic inflow profile
  v_in(y) = 4 U_m y (H - y) / H^2,   U_m = 0.3,  H = 0.41
giving mean inflow velocity Ubar = (2/3) U_m and
  Re = Ubar * D / nu = 20  with  nu = 1e-3.

Reported quantities (benchmark definitions, Schafer & Turek 1996):
  c_D = 2 F_D / (rho Ubar^2 D),   c_L = 2 F_L / (rho Ubar^2 D)
  Delta p = p(front of cylinder) - p(back of cylinder)
  (F_D, F_L) = integral over the cylinder boundary of sigma.n ds,
  sigma = -p I + 2 mu eps(v)

Published benchmark intervals (2D-1, steady, Re=20), Schafer & Turek
(1996), Table 1 (range spanned by the participating groups):
  c_D in [5.5700, 5.5900],  c_L in [0.0104, 0.0110],
  Delta p in [0.1172, 0.1176]

Reference:
  M. Schafer, S. Turek, F. Durst, E. Krause, R. Rannacher, "Benchmark
  Computations of Laminar Flow Around a Cylinder," in Flow Simulation
  with High-Performance Computers II, Notes on Numerical Fluid
  Mechanics vol. 48, Vieweg+Teubner, 1996, pp. 547-566,
  DOI: 10.1007/978-3-322-89849-4_39.
"""
import numpy as np
from skfem import *
from skfem.io.meshio import from_file
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence
from skfem.helpers import dot, grad, sym_grad
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve
import subprocess
import sys

nu = 1.0e-3
Um, H, D = 0.3, 0.41, 0.1
Ubar = 2.0 / 3.0 * Um


def inflow(x):
    y = x[1]
    return np.array([4 * Um * y * (H - y) / H**2, 0 * y])


def run_case(lc_far, lc_cyl, meshfile):
    from dfg_geometry import build_mesh
    build_mesh(meshfile, lc_far=lc_far, lc_cyl=lc_cyl)
    mesh = from_file(meshfile, out=[])

    Ev = ElementVector(ElementTriP2())
    Ep = ElementTriP1()
    bv = Basis(mesh, Ev, intorder=4)
    bp = Basis(mesh, Ep, intorder=4)

    A = nu * vector_laplace.assemble(bv)
    B = divergence.assemble(bv, bp)

    Dv_wall = bv.get_dofs(mesh.boundaries["walls"])
    Dv_cyl = bv.get_dofs(mesh.boundaries["cylinder"])
    Dv_in = bv.get_dofs(mesh.boundaries["inlet"])

    v_bc = bv.project(inflow)
    zero_bc_dofs = np.concatenate([Dv_wall.flatten(), Dv_cyl.flatten()])
    v_bc[zero_bc_dofs] = 0.0
    fixed = np.concatenate([Dv_in.flatten(), zero_bc_dofs])
    free = np.setdiff1d(np.arange(bv.N), fixed)
    free_full = np.concatenate([free, bv.N + np.arange(bp.N)])

    v_h = v_bc.copy()
    for it in range(30):
        v_interp = bv.interpolate(v_h)

        @BilinearForm
        def conv(u, w_test, w):
            velx, vely = w["vel"][0], w["vel"][1]
            gu = grad(u)
            return dot(velx * gu[:, 0] + vely * gu[:, 1], w_test)

        N = conv.assemble(bv, vel=v_interp)
        K = bmat([[A + N, -B.T], [-B, None]], "csc")
        x0 = np.zeros(K.shape[0])
        x0[fixed] = v_bc[fixed]
        rhs_c = -K @ x0
        x = x0.copy()
        x[free_full] = spsolve(K[free_full][:, free_full], rhs_c[free_full])
        v_new = x[:bv.N]
        diff = np.linalg.norm(v_new - v_h) / (np.linalg.norm(v_new) + 1e-14)
        v_h = v_new
        if diff < 1e-10:
            break
    p_h = x[bv.N:]

    fb_cyl = FacetBasis(mesh, Ev, facets=mesh.boundaries["cylinder"], intorder=4)
    fb_cyl_p = FacetBasis(mesh, Ep, facets=mesh.boundaries["cylinder"], intorder=4)

    @Functional
    def drag_integrand(w):
        n = w.n
        eps = sym_grad(w["vh"])
        return -(-w["ph"] * n[0] + 2 * nu * (eps[0][0]*n[0] + eps[0][1]*n[1]))

    @Functional
    def lift_integrand(w):
        n = w.n
        eps = sym_grad(w["vh"])
        return -(-w["ph"] * n[1] + 2 * nu * (eps[1][0]*n[0] + eps[1][1]*n[1]))

    vh_cyl = fb_cyl.interpolate(v_h)
    ph_cyl = fb_cyl_p.interpolate(p_h)
    FD = drag_integrand.assemble(fb_cyl, vh=vh_cyl, ph=ph_cyl)
    FL = lift_integrand.assemble(fb_cyl, vh=vh_cyl, ph=ph_cyl)
    cD = 2 * FD / (Ubar**2 * D)
    cL = 2 * FL / (Ubar**2 * D)

    p_front = bp.probes(np.array([[0.15], [0.2]])) @ p_h
    p_back = bp.probes(np.array([[0.25], [0.2]])) @ p_h
    dp = p_front[0] - p_back[0]

    return bv.N + bp.N, it + 1, diff, cD, cL, dp


print(f"{'mesh':>10} {'ndof':>8} {'Picard it':>10} {'final res':>11} "
      f"{'c_D':>10} {'c_L':>10} {'dP':>10}")
print(f"{'':>10} {'':>8} {'':>10} {'':>11} "
      f"{'[5.57,5.59]':>10} {'[0.0104,0.0110]':>15} {'[0.1172,0.1176]':>15}")
for label, (lc_far, lc_cyl) in [("coarse", (0.035, 0.010)),
                                  ("medium", (0.020, 0.006)),
                                  ("fine", (0.012, 0.0035))]:
    ndof, nit, res, cD, cL, dp = run_case(lc_far, lc_cyl, f"dfg_{label}.msh")
    print(f"{label:>10} {ndof:8d} {nit:10d} {res:11.2e} "
          f"{cD:10.5f} {cL:10.5f} {dp:10.5f}")
