"""
Capstone Project 1 (Case Study 1): Genuine RT x DG mixed Poisson on
unstructured triangular meshes, via scikit-fem's built-in Raviart-Thomas
elements (real Piola-mapped H(div) basis functions, assembled with the
package's own quadrature and sparse assembly -- not a finite-volume /
MAC-scheme stand-in).

Manufactured solution on the unit square:
    p(x,y) = sin(pi x) sin(pi y),      K = I
    u = -grad p
    f = div u = 2 pi^2 sin(pi x) sin(pi y)
Homogeneous Dirichlet data p = 0 on the whole boundary (consistent with
the manufactured p above), imposed *naturally* through the RT boundary
flux term, exactly as derived in the mixed weak form of Chapter 1.

Reference for the element pair and its approximation theory:
  P.-A. Raviart and J.-M. Thomas, "A mixed finite element method for
  second order elliptic problems," Lecture Notes in Math. 606, Springer,
  1977.
  D. Boffi, F. Brezzi, M. Fortin, "Mixed Finite Element Methods and
  Applications," Springer, 2013, Sec. 2.3 (RT spaces) and Ch. 7
  (implementation and numerical tests).
Software pattern adapted from the official scikit-fem example ex37.py
(3D RT1-P0 mixed Poisson demo), reduced to 2D and extended with a
convergence/conservation study and a primal-FEM comparison as required
by this handbook.
"""
import numpy as np
from skfem import *
from skfem.helpers import dot, div, grad


def p_exact(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def f_source(x):
    return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


@BilinearForm
def bilinf(sigma, u, tau, v, w):
    # sigma,tau: flux trial/test (RT); u,v: pressure trial/test (DG)
    return dot(sigma, tau) - div(sigma) * v - div(tau) * u


@LinearForm
def linf(tau, v, w):
    return -f_source(w.x) * v


def solve_mixed(nrefs, order=0):
    mesh = MeshTri().refined(nrefs)
    if order == 0:
        e = ElementTriRT0() * ElementTriP0()
    else:
        # NOTE: in scikit-fem, ElementTriRT0 and ElementTriRT1 are both
        # aliases for the classical lowest-order RT element; the next
        # compatible (quadratic-flux) pair is ElementTriRT2, which must
        # be paired with a *discontinuous* linear pressure space.
        e = ElementTriRT2() * ElementDG(ElementTriP1())
    basis = Basis(mesh, e)

    A = bilinf.assemble(basis)
    b = linf.assemble(basis)
    x = solve(A, b)

    (sigma, sigma_basis), (u, u_basis) = basis.split(x)

    # L2 error of pressure via skfem's built-in error norm utility
    @Functional
    def perr(w):
        return (w["uh"] - p_exact(w.x)) ** 2

    L2_p = np.sqrt(perr.assemble(u_basis, uh=u_basis.interpolate(u)))

    # L2 error of flux (compare against exact u = -grad p)
    @Functional
    def uerr(w):
        ex = np.array([-np.pi*np.cos(np.pi*w.x[0])*np.sin(np.pi*w.x[1]),
                       -np.pi*np.sin(np.pi*w.x[0])*np.cos(np.pi*w.x[1])])
        return (w["sigh"][0] - ex[0])**2 + (w["sigh"][1] - ex[1])**2

    L2_u = np.sqrt(uerr.assemble(sigma_basis, sigh=sigma_basis.interpolate(sigma)))

    # element-wise conservation residual measured two different ways:
    # (1) strong residual against the smooth f (decays at the same rate
    #     as the other errors -- NOT expected to vanish);
    # (2) residual against Pi_h f, the L2 projection of f onto the
    #     discrete pressure space -- THIS is the quantity the weak
    #     equation (div u_h, q) = (f, q) actually forces to zero, and it
    #     should vanish to solver tolerance regardless of mesh size.
    @Functional
    def div_residual(w):
        return (div(w["sigh"]) - f_source(w.x)) ** 2

    cons_L2 = np.sqrt(div_residual.assemble(sigma_basis,
                                             sigh=sigma_basis.interpolate(sigma)))

    f_proj = u_basis.project(f_source)

    @Functional
    def div_residual_proj(w):
        return (w["divsig"] - w["fproj"]) ** 2

    cons_exact = np.sqrt(div_residual_proj.assemble(
        sigma_basis,
        divsig=div(sigma_basis.interpolate(sigma)),
        fproj=u_basis.interpolate(f_proj)))

    ndof = basis.N
    h = mesh.param()
    return h, ndof, L2_p, L2_u, cons_L2, cons_exact


print(f"{'refs':>5} {'h':>10} {'ndof':>7} {'L2(p) err':>12} {'L2(u) err':>12} "
      f"{'||div u_h-f||':>14} {'||div u_h-Pi f||':>17}   rate(p)  rate(u)")
prev = {}
for order, label in [(0, "RT0-DG0 (lowest order)"), (1, "RT2-DG1 (quadratic flux / linear pressure)")]:
    print(f"--- {label} ---")
    prev_p, prev_u = None, None
    for nrefs in [2, 3, 4, 5]:
        h, ndof, L2_p, L2_u, cons, cons_exact = solve_mixed(nrefs, order=order)
        rp = "" if prev_p is None else f"{np.log2(prev_p/L2_p):6.2f}"
        ru = "" if prev_u is None else f"{np.log2(prev_u/L2_u):6.2f}"
        print(f"{nrefs:5d} {h:10.5f} {ndof:7d} {L2_p:12.5e} {L2_u:12.5e} "
              f"{cons:14.5e} {cons_exact:17.5e}  {rp:>7}  {ru:>7}")
        prev_p, prev_u = L2_p, L2_u

print()
print("=== Primal continuous-Galerkin P1 comparison (for Chapter 2's ===")
print("=== primal-vs-mixed discussion) ===")


def solve_primal(nrefs):
    mesh = MeshTri().refined(nrefs)
    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def a(u, v, w):
        return dot(grad(u), grad(v))

    @LinearForm
    def L(v, w):
        return f_source(w.x) * v

    A = a.assemble(basis)
    b = L.assemble(basis)
    D = basis.get_dofs(mesh.boundary_facets())
    Acon, bcon, x0, I = condense(A, b, D=D)
    x = solve(Acon, bcon)
    xf = np.zeros(A.N if hasattr(A, "N") else basis.N)
    xf[I] = x

    @Functional
    def perr(w):
        return (w["uh"] - p_exact(w.x)) ** 2

    L2_p = np.sqrt(perr.assemble(basis, uh=basis.interpolate(xf)))
    return mesh.param(), L2_p


print(f"{'refs':>5} {'h':>10} {'L2(p) err, primal P1':>22}  rate")
prev = None
for nrefs in [2, 3, 4, 5]:
    h, L2p = solve_primal(nrefs)
    rate = "" if prev is None else f"{np.log2(prev/L2p):.2f}"
    print(f"{nrefs:5d} {h:10.5f} {L2p:22.5e}  {rate}")
    prev = L2p

print()
print("Flux (gradient) recovery from the primal solution is only")
print("piecewise-constant per cell (P1 gradient) and is NOT continuous")
print("in the normal direction across cell edges -- unlike the RT flux")
print("above, which is continuous by construction. This is the concrete")
print("meaning of 'primal recovered flux is inaccurate and non-conservative'.")
