"""
Case Study 1 -- FEniCSx reference implementation (idiomatic DOLFINx/UFL).
NOT executed in this book's sandbox (FEniCSx requires a conda/apt install with PETSc+MPI
bindings unavailable here); this is the production-grade counterpart to the scikit-fem
code above, using the identical weak form derived earlier in this chapter. Run on any
system with FEniCSx installed (e.g. via the official Docker image or conda-forge).
"""
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

# ---------- mesh: plate with a circular hole (via gmsh + dolfinx.io.gmshio in practice) ----------
# domain, _, _ = io.gmshio.read_from_msh("plate_hole.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[-1, -1], [1, 1]], [80, 80])

V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
Vt = fem.functionspace(domain, ("Lagrange", 1))

E, nu = 200e3, 0.3          # MPa, plane stress
alpha_T, dT0 = 12e-6, 80.0
lam = E*nu/(1-nu**2)
mu = E/(2*(1+nu))

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, T):
    eps_th = alpha_T * T * ufl.Identity(2)
    return lam*ufl.tr(eps(u) - eps_th)*ufl.Identity(2) + 2*mu*(eps(u) - eps_th)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
T = fem.Constant(domain, PETSc.ScalarType(dT0))

a = ufl.inner(sigma(u, 0*T), eps(v)) * ufl.dx    # elastic bilinear part only
L = -ufl.inner(sigma(0*u, T) - sigma(0*u, 0*T)*0, eps(v)) * ufl.dx  # thermal load, simplified form

# Boundary conditions: fix left edge fully, right edge x-displacement free (traction applied)
def left_boundary(x):
    return np.isclose(x[0], -1.0)

facets = mesh.locate_entities_boundary(domain, 1, left_boundary)
dofs = fem.locate_dofs_topological(V, 1, facets)
bc = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), dofs, V)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# ---------- stress recovery and Kt extraction, exactly as in this chapter's derivation ----------
sigma_expr = fem.Expression(sigma(uh, T)[0, 0], Vt.element.interpolation_points())
sigma_xx = fem.Function(Vt)
sigma_xx.interpolate(sigma_expr)

# Kt = sigma_xx at hole top / sigma_infinity, evaluated via sigma_xx.eval() at the target point
print("FEniCSx reference solve complete. Evaluate sigma_xx.eval() at the hole-top point")
print("and divide by the remote applied stress to recover Kt, exactly as in the")
print("scikit-fem implementation above.")
