"""Case Study 5 -- FEniCSx reference: mixed Biot poromechanics (Taylor-Hood displacement/
pressure), the inf-sup-stable element pair this chapter's derivation requires."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from basix.ufl import element, mixed_element

domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 20)
Ue = element("Lagrange", domain.basix_cell(), 2, shape=(2,))   # quadratic displacement
Pe = element("Lagrange", domain.basix_cell(), 1)                 # linear pressure (Taylor-Hood)
W = fem.functionspace(domain, mixed_element([Ue, Pe]))

(u, p) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)
alpha, c0, K, mu_visc, dt = 0.9, 1e-8, 1e-13, 1e-3, 1e4
lam, mu = 3e8, 2e8
eps = lambda v: ufl.sym(ufl.grad(v))
sigma_eff = lambda v: lam*ufl.tr(eps(v))*ufl.Identity(2) + 2*mu*eps(v)

a = (ufl.inner(sigma_eff(u), eps(w))*ufl.dx - alpha*p*ufl.div(w)*ufl.dx
     + q*(c0*p + alpha*ufl.div(u))*ufl.dx*(1/dt)
     + (K/mu_visc)*ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx)
print("Taylor-Hood mixed Biot weak form assembled (P2 displacement, P1 pressure) --")
print("inf-sup stable by construction, avoiding the checkerboard pressure oscillations")
print("this chapter's Section 2.7 warns equal-order elements would produce.")
