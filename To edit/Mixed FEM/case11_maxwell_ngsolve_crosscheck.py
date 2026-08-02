"""
Cross-validation of Case Study 11 (Maxwell cavity) using NGSolve's
native HCurl (Nedelec) space, independent of the scikit-fem
implementation in the main capstone. Agreement between two completely
independent FEM codes on the same eigenvalue problem is strong evidence
against an implementation-specific bug in either one.
"""
import ngsolve as ng
from netgen.geom2d import unit_square
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.08))
fes = ng.HCurl(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()

a = ng.BilinearForm(ng.curl(u)*ng.curl(v)*ng.dx).Assemble()
m = ng.BilinearForm(u*v*ng.dx).Assemble()

free = np.array([i for i in range(fes.ndof) if fes.FreeDofs()[i]])

rows_a, cols_a, vals_a = a.mat.COO()
rows_m, cols_m, vals_m = m.mat.COO()
A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(fes.ndof, fes.ndof)).tocsr()
M = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(fes.ndof, fes.ndof)).tocsr()

Af = A[free][:, free].toarray()
Mf = M[free][:, free].toarray()

vals = eigh(Af, Mf, eigvals_only=True)
vals = np.sort(vals[vals > 1e-6])

print("NGSolve HCurl (lowest order) cavity eigenvalues:",
      [f"{v:.4f}" for v in vals[:6]])
analytical = sorted(set(round(np.pi**2*(m1**2+n1**2), 4)
                    for m1 in range(4) for n1 in range(4) if not (m1 == 0 and n1 == 0)))[:6]
print("Analytical pi^2(m^2+n^2):", analytical)
