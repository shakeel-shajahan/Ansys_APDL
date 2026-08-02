"""
Capstone Project 1 (Case Study 1): Steady confined-aquifer flow to a
pumping well -- the mixed-FEM analogue of the classical Thiem (1906)
well-hydraulics problem, still the standard hand-calculation used
worldwide to interpret aquifer pumping tests. Genuine RT-type
(H(div)) mixed Poisson, executed with NGSolve.

Physical setup: an annular confined aquifer r_w < r < R, transmissivity
T, pumped at a constant rate Q through the inner (well) boundary.
Darcy/groundwater flux u = -T grad(h) is the ACTUAL engineering
quantity of interest in a pump test (it IS the pumping rate the
engineer controls and measures), exactly the "flux must be
approximated directly" motivation for mixed FEM from Chapter 1 -- not a
manufactured mathematical convenience.

Boundary conditions (Chapter 1's essential/natural role-reversal,
concretely): the pumping rate Q is prescribed as an ESSENTIAL condition
on the H(div) flux space at the well boundary (a fixed, known extraction
rate -- exactly what a well operator controls); the far-field head H0
is prescribed as a NATURAL condition at the outer boundary.

Validation: the classical Thiem (1906) steady-radial-flow solution,
  h(r) = H0 - Q/(2 pi T) * ln(R/r),
still the industry-standard formula taught in every hydrogeology course
and used directly to estimate aquifer transmissivity from a pumping
test's drawdown data.

Reference:
  G. Thiem, "Hydrologische Methoden," Gebhardt, Leipzig, 1906 (origin
  of the steady radial-flow well equation still in universal use).
  C.V. Theis, "The relation between the lowering of the piezometric
  surface and the rate and duration of discharge of a well using
  ground-water storage," Trans. Amer. Geophys. Union 16(2), 1935,
  pp. 519-524 (the transient generalization; a natural extension task).
"""
import ngsolve as ng
from netgen.geom2d import SplineGeometry
import numpy as np
import math

rw, R = 0.1, 10.0     # well radius, aquifer influence radius [m]
Q, T = 1.0, 1.0        # pumping rate [m^2/s per unit thickness], transmissivity [m^2/s]
H0 = 0.0               # far-field head [m]


def build_mesh(maxh):
    geo = SplineGeometry()
    geo.AddCircle((0, 0), R, bc="outer", leftdomain=1, rightdomain=0)
    geo.AddCircle((0, 0), rw, bc="well", leftdomain=0, rightdomain=1)
    mesh = ng.Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(3)
    return mesh


def solve(maxh, order=2):
    mesh = build_mesh(maxh)
    fesU = ng.HDiv(mesh, order=order, dirichlet="well")
    fesP = ng.L2(mesh, order=order - 1)
    fes = ng.FESpace([fesU, fesP])
    (u, p), (v, q) = fes.TnT()

    a = ng.BilinearForm(fes)
    a += ((1.0/T) * u * v - ng.div(u) * q - ng.div(v) * p) * ng.dx
    a.Assemble()

    f = ng.LinearForm(fes)
    f.Assemble()

    gfu = ng.GridFunction(fes)
    n = ng.specialcf.normal(2)
    qflux = Q / (2 * np.pi * rw)
    gfu.components[0].Set(-qflux * n, definedon=mesh.Boundaries("well"))

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    inv = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack")
    gfu.vec.data += inv * res

    p_h = gfu.components[1]

    thiem_cf = ng.CoefficientFunction(
        H0 - Q / (2 * np.pi * T) * ng.log(R / ng.sqrt(ng.x**2 + ng.y**2 + 1e-30)))
    err = ng.sqrt(ng.Integrate((p_h - thiem_cf) ** 2 * ng.dx, mesh))

    u_h = gfu.components[0]
    flux_well = ng.Integrate(-u_h * n * ng.ds(definedon=mesh.Boundaries("well")), mesh)

    # exact flux from the Thiem solution: u = -T grad(h) = (Q/(2 pi r)) r_hat
    rr = ng.sqrt(ng.x**2 + ng.y**2 + 1e-30)
    u_exact_cf = ng.CoefficientFunction((Q/(2*np.pi)*ng.x/rr**2, Q/(2*np.pi)*ng.y/rr**2))
    flux_err = ng.sqrt(ng.Integrate(
        ((u_h[0]-u_exact_cf[0])**2 + (u_h[1]-u_exact_cf[1])**2) * ng.dx, mesh))

    # cellwise conservation: for every element K, |int_{dK} u_h.n ds| should be
    # exactly zero away from the well (no source term f=0 anywhere in this domain)
    fesP0 = ng.L2(mesh, order=0)
    cellwise_residual = ng.GridFunction(fesP0)
    cellwise_residual.Set(ng.div(u_h))
    max_cellwise = max(abs(cellwise_residual.vec[i]) for i in range(len(cellwise_residual.vec)))

    return mesh, fes, gfu, mesh.nv, err, flux_well, flux_err, max_cellwise


print(f"{'maxh':>8} {'nv':>7} {'L2(h) err vs Thiem':>20} {'rate':>7} "
      f"{'well flux (should = Q=1.0)':>28} {'L2(u) flux err':>15} "
      f"{'max cellwise |div u_h|':>24}")
prev = None
last = None
for maxh in [0.8, 0.4, 0.2, 0.1]:
    mesh, fes, gfu, nv, err, flux_well, flux_err, max_cellwise = solve(maxh)
    rate = "" if prev is None else f"{math.log2(prev/err):.2f}"
    print(f"{maxh:8.2f} {nv:7d} {err:20.6e} {rate:>7} {flux_well:28.6f} "
          f"{flux_err:15.6e} {max_cellwise:24.3e}")
    prev = err
    last = (mesh, gfu)

print()
print("Assumptions behind the 'optimal convergence' claim made in the")
print("chapter text: (1) the manufactured/analytical comparison (Thiem)")
print("is itself exact only in the STEADY, radially-symmetric, homogeneous")
print("-transmissivity idealization -- real aquifers are rarely perfectly")
print("radially symmetric or homogeneous; (2) 'optimal' here means the")
print("rate matches the classical a priori estimate for this element pair")
print("on a SMOOTH solution -- the Thiem solution has a logarithmic")
print("singularity exactly at the well, so the rate reported is the")
print("rate observed on this specific annular domain (which excludes")
print("the singular point itself), not a rate valid all the way to r=0.")

print()
print("Point-sample comparison at the finest mesh (maxh=0.1):")
mesh, gfu = last
p_h = gfu.components[1]
print(f"{'r [m]':>8} {'FEM head [m]':>14} {'Thiem head [m]':>16} {'abs diff':>12}")
for r in [0.2, 0.5, 1.0, 2.0, 5.0, 9.0]:
    val = p_h(mesh(r, 0.0))
    th = H0 - Q / (2 * np.pi * T) * math.log(R / r)
    print(f"{r:8.2f} {val:14.6f} {th:16.6f} {abs(val-th):12.3e}")
