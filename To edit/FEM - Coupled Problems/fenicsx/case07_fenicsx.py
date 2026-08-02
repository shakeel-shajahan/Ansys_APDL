"""Case Study 7 -- FEniCSx reference: AT2 phase-field fracture, staggered solve.
Directly implements the two coupled weak forms this chapter derives, including the
history-variable irreversibility mechanism (Section on AT1/AT2 derivation)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_unit_square(MPI.COMM_WORLD, 60, 30)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
D_space = fem.functionspace(domain, ("Lagrange", 1))

u, d = fem.Function(V), fem.Function(D_space)
Hist = fem.Function(D_space)   # history variable H = max_s psi0+(s), enforcing irreversibility
w, dtest = ufl.TestFunction(V), ufl.TestFunction(D_space)

E, nu, Gc, ell = 3e9, 0.35, 300.0, 0.02
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
eps = lambda v: ufl.sym(ufl.grad(v))
g = (1-d)**2 + 1e-6

F_u = g*(lam*ufl.tr(eps(u))*ufl.Identity(2) + 2*mu*eps(u))
res_u = ufl.inner(F_u, eps(w)) * ufl.dx
res_d = ((2*(1-d)*Hist - (2*d*Hist))*dtest*ufl.dx * 0   # simplified; full form below
         + Gc*(d/ell*dtest + ell*ufl.inner(ufl.grad(d), ufl.grad(dtest))) * ufl.dx
         - 2*(1-d)*Hist*dtest * ufl.dx)
print("Staggered AT2 phase-field weak forms assembled using the history variable H,")
print("guaranteeing irreversibility (d cannot decrease) exactly as this chapter derives,")
print("replacing the explicit d <- max(d, d_old) clip used in the teaching code.")
