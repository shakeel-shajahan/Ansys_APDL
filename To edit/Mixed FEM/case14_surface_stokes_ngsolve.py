"""
Capstone Project 14 (Case Study 14): Genuine surface Stokes flow on a
real curved sphere, executed with NGSolve (which, unlike scikit-fem in
this environment, natively supports 3D-embedded 2D manifold meshes and
surface differential operators). This directly replaces the "not
executed" status of Case Study 14 in the previous edition of this
handbook.

Domain: unit sphere surface, embedded in R^3, meshed with Netgen's CSG
kernel (real curved boundary, not a flat approximation at the element
level -- NGSolve uses curved (isoparametric) surface elements).

Velocity: VectorH1 (3 Cartesian components) restricted to the sphere
boundary, with a tangentiality PENALTY (1/eta)(u.n, w.n) -- see Case
Study 14's chapter text for the eta-vs-conditioning trade-off this
capstone verifies directly. Pressure: H1 on the same surface, enforcing
surface incompressibility div_Gamma(u) = 0.

Verification strategy (since a closed-form manufactured solution on a
curved surface is analytically demanding): rather than a convergence
-rate study against an exact solution, this capstone verifies the TWO
STRUCTURAL PROPERTIES a correct surface Stokes solve must have for an
arbitrary tangential forcing: (a) the computed velocity's normal
component vanishes as the penalty eta -> 0, and (b) its surface
divergence is small, both reported quantitatively across a penalty
sweep, exactly the trade-off Problem 14.1 (in the chapter text)
describes.

Reference:
  A. Reusken, "Analysis of the Taylor-Hood surface finite element
  method for the surface Stokes equation," 2024.
  A. Demlow and M. Neilan, "A Taylor-Hood finite element method for
  the surface Stokes problem without penalization," 2025.
  M. Nestler, I. Nitschke, A. Voigt, "A finite element approach for
  vector- and tensor-valued surface PDEs," J. Comput. Phys. 389, 2019,
  pp. 48-61 (the general surface-FEM machinery this capstone follows).
"""
import ngsolve as ng
from netgen.csg import CSGeometry, Sphere, Pnt
import numpy as np


def build_mesh(maxh):
    geo = CSGeometry()
    geo.Add(Sphere(Pnt(0, 0, 0), 1.0).bc("sphere"))
    mesh = ng.Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(3)   # genuine curved (order-3 isoparametric) surface elements
    return mesh


mesh = build_mesh(0.25)
bnd = mesh.Boundaries("sphere")
n = ng.specialcf.normal(3)


def surf_grad(u):
    gu = ng.grad(u).Trace()
    return gu - ng.OuterProduct(gu * n, n)


def solve_surface_stokes(eta, order=2, mesh=mesh):
    bnd = mesh.Boundaries("sphere")
    n = ng.specialcf.normal(3)

    def surf_grad_local(u):
        gu = ng.grad(u).Trace()
        return gu - ng.OuterProduct(gu * n, n)

    Vh = ng.VectorH1(mesh, order=order, definedon=bnd)
    Qh = ng.H1(mesh, order=order - 1, definedon=bnd)
    X = ng.FESpace([Vh, Qh])
    (u, p), (v, q) = X.TnT()

    mu = 1.0

    def eps_surf(w):
        gw = surf_grad_local(w)
        return 0.5 * (gw + gw.trans)

    a = ng.BilinearForm(X)
    a += (2*mu*ng.InnerProduct(eps_surf(u), eps_surf(v))
          - p*ng.Trace(surf_grad_local(v)) + q*ng.Trace(surf_grad_local(u))
          + (1.0/eta)*(u*n)*(v*n)) * ng.ds
    a.Assemble()

    # tangential forcing: a smooth "wind" field projected tangent to the
    # sphere, f = (-y, x, 0) (rotation about the z-axis), already tangent
    f = ng.LinearForm(X)
    wind = ng.CoefficientFunction((-ng.y, ng.x, 0.0))
    f += (wind * v) * ng.ds
    f.Assemble()

    gfu = ng.GridFunction(X)
    inv = a.mat.Inverse(X.FreeDofs())
    gfu.vec.data = inv * f.vec

    uh = gfu.components[0]

    # verification (a): normal-component (tangentiality violation) norm
    un_sq = ng.Integrate((uh*n)**2 * ng.ds, mesh)
    u_sq = ng.Integrate((uh[0]**2+uh[1]**2+uh[2]**2) * ng.ds, mesh)
    tangentiality_error = np.sqrt(un_sq / max(u_sq, 1e-300))

    # verification (b): surface divergence norm (incompressibility)
    divu_sq = ng.Integrate(ng.Trace(surf_grad_local(uh))**2 * ng.ds, mesh)
    surf_div_norm = np.sqrt(divu_sq)

    ndof = X.ndof
    return ndof, tangentiality_error, surf_div_norm


print(f"{'eta':>10} {'ndof':>7} {'||u.n||/||u||':>16} {'||div_Gamma u||':>18}")
for eta in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]:
    ndof, tang_err, div_norm = solve_surface_stokes(eta)
    print(f"{eta:10.1e} {ndof:7d} {tang_err:16.6e} {div_norm:18.6e}")

print()
print("Separate diagnostic: genuine MESH-REFINEMENT study at FIXED eta,")
print("isolating discretization error from penalty error (a penalty sweep")
print("alone, as printed above, cannot separate the two).")
print(f"{'maxh':>8} {'ndof':>7} {'||u.n||/||u||':>16} {'||div_Gamma u||':>18}")
eta_fixed = 1e-3
for maxh in [0.5, 0.35, 0.25, 0.18]:
    mesh_r = build_mesh(maxh)
    ndof, tang_err, div_norm = solve_surface_stokes(eta_fixed, mesh=mesh_r)
    print(f"{maxh:8.2f} {ndof:7d} {tang_err:16.6e} {div_norm:18.6e}")
