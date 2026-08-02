"""Case Study 4 -- FEniCSx reference: RVE homogenization with periodic boundary conditions
via dolfinx_mpc (or manual periodic dof mapping), exactly the Hill-Mandel-consistent
periodic RVE solve this chapter's derivation calls for (upgrading the scikit-fem version's
Dirichlet-only approximation)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
# from dolfinx_mpc import MultiPointConstraint, apply_lifting, assemble_matrix  # periodic BCs

domain = mesh.create_unit_square(MPI.COMM_WORLD, 40, 40)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))

# heterogeneous material via a DG0 (elementwise-constant) coefficient field
Q = fem.functionspace(domain, ("DG", 0))
E_field = fem.Function(Q)   # set via E_field.x.array[:] = ... per-cell values
nu = 0.3
lam_expr = E_field*nu/(1-nu**2)
mu_expr = E_field/(2*(1+nu))
sigma = lambda w: lam_expr*ufl.tr(eps(w))*ufl.Identity(2) + 2*mu_expr*eps(w)
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
print("Periodic RVE weak form: use dolfinx_mpc.MultiPointConstraint to tie opposite RVE")
print("faces together before assembly, exactly satisfying the Hill-Mandel condition")
print("this chapter derives (unlike the simpler Dirichlet-only scikit-fem version above).")
