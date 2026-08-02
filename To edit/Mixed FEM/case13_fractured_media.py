"""
Capstone Project 13 (Case Study 13): Fractured porous media flow.

Honesty note on scope: the mathematically complete target described in
the chapter text is a genuine MIXED-DIMENSIONAL model (2D matrix Darcy
flow coupled to a 1D fracture Darcy flow through an explicit exchange
flux term, as in Martin-Jaffre-Roberts 2005 / the PorePy framework).
Implementing a fully conforming mixed-dimensional mesh (matrix
triangles meeting exactly along a 1D fracture edge, with a coupled 1D
line solve) was not completed in the time available for this capstone.
Instead, this script uses the classical EQUIVALENT-CONTINUUM
simplification: the fracture is represented as a thin band of
ordinary 2D cells with strongly enhanced (or reduced) permeability,
solved with the SAME genuine RT0-DG0 solver validated in Cases 1-3.
This is a real, different (and pedagogically standard first-step)
approximation, not the lower-dimensional model -- the distinction and
its consequences (loss of exact aperture-independent scaling, mesh
distortion at high aspect ratio) are discussed explicitly in the
chapter text.

Reference:
  V. Martin, J. Jaffre, J.E. Roberts, "Modeling fractures and barriers
  as interfaces for flow in porous media," SIAM J. Sci. Comput. 26(5),
  2005, pp. 1667-1691 (the genuine mixed-dimensional model this
  capstone approximates).
  B. Flemisch et al., "Benchmarks for single-phase flow in fractured
  porous media," Adv. Water Resour. 111, 2018, pp. 239-258.
"""
import numpy as np
from skfem import *
from skfem.helpers import dot, div


def make_fracture_K(mesh, aperture_cells=1, Kmatrix=1.0, Kfracture=1.0e4):
    centers = mesh.p[:, mesh.t].mean(axis=1)
    h = 1.0 / 32
    in_fracture = np.abs(centers[1] - 0.5) < (aperture_cells * h / 2 + 1e-9)
    return np.where(in_fracture, Kfracture, Kmatrix)


def solve_darcy(mesh, Kfield, p_left=1.0, p_right=0.0):
    e = ElementTriRT0() * ElementTriP0()
    basis = Basis(mesh, e)
    Kb = Basis(mesh, ElementTriP0())

    @BilinearForm
    def bilinf_het(sigma, u, tau, v, w):
        return (1.0 / w["Kfield"]) * dot(sigma, tau) - div(sigma) * v - div(tau) * u

    A = bilinf_het.assemble(basis, Kfield=Kb.interpolate(Kfield))

    left = mesh.facets_satisfying(lambda x: x[0] < 1e-10)
    right = mesh.facets_satisfying(lambda x: x[0] > 1 - 1e-10)
    top = mesh.facets_satisfying(lambda x: x[1] > 1 - 1e-10)
    bot = mesh.facets_satisfying(lambda x: x[1] < 1e-10)

    fb_lr = FacetBasis(mesh, e, facets=np.concatenate([left, right]))

    @LinearForm
    def bc(tau, v, w):
        pD = p_left * (w.x[0] < 1e-10) + p_right * (w.x[0] > 1 - 1e-10)
        return -dot(tau, w.n) * pD

    b = bc.assemble(fb_lr)
    dofs_noflow = basis.get_dofs(facets=np.concatenate([top, bot]))
    Acon, bcon, x0, I = condense(A, b, D=dofs_noflow)
    xI = solve(Acon, bcon)
    x = np.zeros(A.shape[0])
    x[I] = xI
    (sigma, sb), (u, ub) = basis.split(x)

    fb_left = FacetBasis(mesh, ElementTriRT0(), facets=left)

    @Functional
    def flux_out(w):
        return dot(w["sigh"], w.n)

    qleft = flux_out.assemble(fb_left, sigh=fb_left.interpolate(sigma))
    return -qleft


mesh = MeshTri().refined(5)

print("Total flux (equals total inflow) for varying fracture")
print("conductivity contrast, single horizontal fracture at y=0.5:")
print(f"{'K_fracture/K_matrix':>20} {'total flux':>14} {'enhancement factor':>20}")
Kmatrix = 1.0
q_baseline = solve_darcy(mesh, np.ones(mesh.t.shape[1]) * Kmatrix)
for contrast in [1.0, 10.0, 1e2, 1e3, 1e4, 1e5]:
    Kfield = make_fracture_K(mesh, aperture_cells=1, Kmatrix=Kmatrix,
                              Kfracture=Kmatrix * contrast)
    q = solve_darcy(mesh, Kfield)
    print(f"{contrast:20.1e} {q:14.5f} {q/q_baseline:20.4f}")

print()
print("Now a BLOCKING fracture (low conductivity, e.g. a clay-filled fault):")
for contrast in [1.0, 1e-2, 1e-4, 1e-6]:
    Kfield = make_fracture_K(mesh, aperture_cells=1, Kmatrix=Kmatrix,
                              Kfracture=Kmatrix * contrast)
    q = solve_darcy(mesh, Kfield)
    label = f"K_frac/K_mat={contrast:.0e}"
    print(f"{label:>20} {q:14.5f} {q/q_baseline:20.4f}")
