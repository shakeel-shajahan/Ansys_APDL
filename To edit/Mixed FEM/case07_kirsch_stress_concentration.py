"""
Capstone Project 7 (Case Study 7): Stress concentration around a
circular hole in a plate under uniaxial tension -- the classical
Kirsch (1898) problem, still the textbook starting point for every
fatigue and fracture design course, and the reason "stress
concentration factor" (SCF) is a universal term in mechanical and
aerospace engineering. Genuine strongly-symmetric Hellinger-Reissner
-type mixed elasticity (the TDNNS method: Tangential-Displacement
Normal-Normal-Stress continuous elements), executed with NGSolve's
native HDivDiv (symmetric H(div,div)-conforming stress) and HCurl
(tangentially-continuous displacement) spaces.

This is a materially DIFFERENT, and mathematically STRONGER,
discretization than the PEERS weak-symmetry method used in an earlier
edition of this handbook's Case Study 7: TDNNS gives a POINTWISE
symmetric stress tensor by construction (no rotation multiplier needed
at all), because NGSolve implements HDivDiv natively, closing the
honesty-note gap this handbook previously carried ("this software does
not implement Arnold-Winther or another strongly-symmetric element").

Geometry: a quarter-plate (0<=x<=W, 0<=y<=W) with a quarter-circle hole
of radius a at the origin, W/a=10 (large enough to approximate an
infinite plate), symmetry boundary conditions on the two straight
edges through the hole, uniaxial tension sigma_inf applied on the far
edge x=W.

Validation: the classical Kirsch (1898) exact solution for an infinite
plate, sigma_theta_theta(a,theta) = sigma_inf*(1 - 2cos(2 theta)),
giving the famous stress concentration factor of 3 at theta=90 degrees
(perpendicular to the load) and -1 at theta=0 (along the load).

Reference:
  E.G. Kirsch, "Die Theorie der Elastizitat und die Bedurfnisse der
  Festigkeitslehre," Zeitschrift des Vereines Deutscher Ingenieure 42,
  1898, pp. 797-807 (origin of the stress-concentration-factor concept
  still taught in every mechanical/aerospace design course).
  A.S. Pechstein and J. Schoberl, "Tangential-displacement and
  normal-normal-stress continuous mixed finite elements for
  elasticity," Math. Models Methods Appl. Sci. 21(8), 2011,
  pp. 1761-1782 (the TDNNS method used here).
"""
import ngsolve as ng
from netgen.geom2d import SplineGeometry
import numpy as np

hole_radius = 1.0
W = 10.0             # W/hole_radius = 10, approximating an infinite plate
sigma_inf = 1.0
E, nu = 1.0, 0.3
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))


def make_mesh(maxh_global, maxh_hole):
    geo = SplineGeometry()
    pts = [geo.AppendPoint(*p) for p in
           [(hole_radius, 0), (W, 0), (W, W), (0, W), (0, hole_radius)]]
    geo.Append(["line", pts[0], pts[1]], bc="bottom", maxh=maxh_global)
    geo.Append(["line", pts[1], pts[2]], bc="right", maxh=maxh_global)
    geo.Append(["line", pts[2], pts[3]], bc="top", maxh=maxh_global)
    geo.Append(["line", pts[3], pts[4]], bc="left", maxh=maxh_global)
    mid = geo.AppendPoint(hole_radius * np.cos(np.pi / 4), hole_radius * np.sin(np.pi / 4))
    geo.Append(["spline3", pts[4], mid, pts[0]], bc="hole", maxh=maxh_hole)
    mesh = ng.Mesh(geo.GenerateMesh(maxh=maxh_global))
    mesh.Curve(3)
    return mesh


def solve(maxh_global, maxh_hole, order):
    mesh = make_mesh(maxh_global, maxh_hole)
    V = ng.HDivDiv(mesh, order=order, dirichlet="hole|top|right")
    Q = ng.HCurl(mesh, order=order, dirichlet="left|bottom")
    X = ng.FESpace([V, Q])
    (sigma, u), (tau, v) = X.TnT()
    n = ng.specialcf.normal(2)

    def tang(w):
        return w - (w * n) * n

    def compliance(s):
        tr = ng.Trace(s)
        return 1 / (2 * mu) * (s - lam / (2 * mu + 2 * lam) * tr * ng.Id(2))

    a = ng.BilinearForm(X, symmetric=True)
    a += (ng.InnerProduct(compliance(sigma), tau) + ng.div(sigma) * v + ng.div(tau) * u) * ng.dx
    a += (-(sigma * n) * tang(v) - (tau * n) * tang(u)) * ng.dx(element_boundary=True)
    a.Assemble()

    f = ng.LinearForm(X)
    f.Assemble()

    gfsol = ng.GridFunction(X)
    gfsol.components[0].Set(sigma_inf * ng.OuterProduct(n, n), definedon=mesh.Boundaries("right"))

    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfsol.vec
    inv = a.mat.Inverse(X.FreeDofs(), inverse="umfpack")
    gfsol.vec.data += inv * res

    sigma_h = gfsol.components[0]
    eps = 1e-3
    scf_tension_side = sigma_h(mesh(eps, hole_radius + eps))[0]
    scf_compression_side = sigma_h(mesh(hole_radius + eps, eps))[3]
    return mesh.nv, scf_tension_side, scf_compression_side


print(f"{'nv':>7} {'maxh_hole':>10} {'order':>6} "
      f"{'sigma_tt at theta=90 [Kirsch: 3.0]':>36} "
      f"{'sigma_tt at theta=0 [Kirsch: -1.0]':>36}")
for maxh_hole, order in [(0.30, 2), (0.15, 2), (0.08, 3), (0.04, 3)]:
    nv, scf90, scf0 = solve(2.0, maxh_hole, order)
    print(f"{nv:7d} {maxh_hole:10.3f} {order:6d} {scf90:36.5f} {scf0:36.5f}")
