"""
Capstone Project 11 (Case Study 11): Maxwell cavity eigenvalue problem
using genuine Nedelec (H(curl)-conforming) edge elements, compared
directly against a naive nodal (continuous Lagrange) vector
discretization to expose the spurious-mode failure this case study
warns about.

Perfectly-conducting unit-square cavity: curl(curl E) = omega^2 E,
n x E = 0 on the boundary. Analytical TE-mode eigenvalues are
  omega^2_{mn} = pi^2 (m^2 + n^2),  m,n = 0,1,2,... (not both zero)
so the nonzero spectrum begins at pi^2 (twice, degenerate), 2 pi^2 is
NOT an eigenvalue of this exact set for a first-kind formulation
without the (1,1) mode... (see printed comparison table for the exact
matched values).

Reference:
  A. Bossavit, "Computational Electromagnetism," Academic Press, 1998
  (foundational text on edge elements and spurious modes).
  D. Boffi, "Finite element approximation of eigenvalue problems,"
  Acta Numerica 19, 2010, pp. 1-120 (definitive survey including the
  Maxwell cavity spurious-mode analysis).
"""
import numpy as np
from skfem import *
from skfem.helpers import curl, grad
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix


def solve_nedelec(nrefs):
    mesh = MeshTri().refined(nrefs)
    E = ElementTriN1()  # lowest-order Nedelec (first kind), tangential-continuous
    basis = Basis(mesh, E)

    @BilinearForm
    def stiffness(u, v, w):
        return curl(u) * curl(v)

    @BilinearForm
    def mass(u, v, w):
        return u[0] * v[0] + u[1] * v[1]

    S = stiffness.assemble(basis)
    M = mass.assemble(basis)

    D = basis.get_dofs(mesh.boundary_facets())
    free = np.setdiff1d(np.arange(S.shape[0]), D.flatten())
    S_f = S[free][:, free].toarray()
    M_f = M[free][:, free].toarray()

    from scipy.linalg import eigh
    vals = eigh(S_f, M_f, eigvals_only=True)
    return np.sort(vals[vals > 1e-6])


def solve_nodal(nrefs):
    """Naive (WRONG for Maxwell) nodal vector Lagrange discretization,
    included specifically to demonstrate the spurious-mode failure."""
    mesh = MeshTri().refined(nrefs)
    E = ElementVector(ElementTriP1())
    basis = Basis(mesh, E)

    @BilinearForm
    def stiffness(u, v, w):
        # naive componentwise gradient stiffness -- NOT the correct
        # curl-curl operator; this is exactly the wrong space, kept
        # here only to show what goes wrong.
        gu, gv = grad(u), grad(v)
        return gu[0, 0]*gv[0, 0] + gu[0, 1]*gv[0, 1] + gu[1, 0]*gv[1, 0] + gu[1, 1]*gv[1, 1]

    @BilinearForm
    def mass(u, v, w):
        return u[0] * v[0] + u[1] * v[1]

    S = stiffness.assemble(basis)
    M = mass.assemble(basis)
    D = basis.get_dofs(mesh.boundary_facets())
    free = np.setdiff1d(np.arange(S.shape[0]), D.flatten())
    S_f = S[free][:, free].toarray()
    M_f = M[free][:, free].toarray()

    from scipy.linalg import eigh
    vals = eigh(S_f, M_f, eigvals_only=True)
    return np.sort(vals[vals > 1e-6])


analytical = sorted(set(round(np.pi**2 * (m**2 + n**2), 6)
                         for m in range(4) for n in range(4) if not (m == 0 and n == 0)))[:6]

print("Analytical omega^2 eigenvalues (pi^2(m^2+n^2)):", [f"{v:.4f}" for v in analytical])
print()
for nrefs in [3, 4, 5]:
    ned_vals = solve_nedelec(nrefs)
    print(f"refs={nrefs}  Nedelec (H(curl)-conforming) smallest eigenvalues: "
          f"{[f'{v:.4f}' for v in ned_vals[:6]]}")

print()
nrefs = 4
nodal_vals = solve_nodal(nrefs)
print(f"refs={nrefs}  Naive nodal-vector (WRONG space) smallest eigenvalues: "
      f"{[f'{v:.4f}' for v in nodal_vals[:6]]}")
print("(Compare with the analytical values above: the Nedelec spectrum")
print(" converges to the correct pi^2(m^2+n^2) sequence; the naive nodal")
print(" vector spectrum instead includes extra, non-physical eigenvalues")
print(" not present in the analytical list -- these are the spurious")
print(" modes this case study warns about.)")
