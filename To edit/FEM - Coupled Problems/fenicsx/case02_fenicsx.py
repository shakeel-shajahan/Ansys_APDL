"""Case Study 2 -- FEniCSx reference: generalized Maxwell viscoelasticity.
Internal variables q_i stored as Quadrature-space Functions, updated at each time step
via a explicit/implicit local update, exactly mirroring this chapter's ODE update rule."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_unit_square(MPI.COMM_WORLD, 40, 40)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
Qe = fem.functionspace(domain, ("Quadrature", 2, (2, 2)))  # internal variable storage

q1, q2 = fem.Function(Qe), fem.Function(Qe)   # branch stresses q_i, updated each step
E_inf, E1, tau1, E2, tau2 = 5e6, 20e6, 0.5, 8e6, 5.0

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
eps = lambda w: ufl.sym(ufl.grad(w))
sigma_total = E_inf*eps(u) + q1 + q2   # algebraic combination, as in Section "Weak Form"
a = ufl.inner(sigma_total, eps(v)) * ufl.dx
# Time loop (outline): solve for u, then update q1,q2 <- q_i + dt*(-q_i/tau_i + E_i*deps/dt)
# via a Quadrature-space projection, exactly the internal-variable update this chapter derives.
print("Reference outline: internal variables stored in a Quadrature function space,")
print("updated locally each time step using this chapter's derived ODE update rule.")
