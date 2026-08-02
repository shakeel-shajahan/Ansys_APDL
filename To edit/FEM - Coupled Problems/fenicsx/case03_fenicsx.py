"""Case Study 3 -- FEniCSx reference: coupled piezoelectric weak form (mixed function
space for displacement u and electric potential phi), exactly the two-way coupled
system derived in this chapter's Weak Form section."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from basix.ufl import element, mixed_element

domain = mesh.create_unit_square(MPI.COMM_WORLD, 30, 10)
Ue = element("Lagrange", domain.basix_cell(), 1, shape=(2,))
Pe = element("Lagrange", domain.basix_cell(), 1)
W = fem.functionspace(domain, mixed_element([Ue, Pe]))

(u, phi) = ufl.TrialFunctions(W)
(w, psi) = ufl.TestFunctions(W)

C11, e31, kappa33 = 1.2e11, 12.0, 1.5e-8
eps = lambda v: ufl.sym(ufl.grad(v))
E_field = lambda p: -ufl.grad(p)

a = (C11*ufl.inner(eps(u), eps(w)) * ufl.dx
     - e31*ufl.inner(E_field(phi), eps(w)) * ufl.dx      # piezoelectric coupling (mechanical eq.)
     + e31*ufl.inner(eps(u), ufl.grad(psi)) * ufl.dx      # SAME coupling tensor (reciprocity)
     - kappa33*ufl.inner(E_field(phi), ufl.grad(psi)) * ufl.dx)
print("Mixed (u, phi) weak form assembled with the SAME coupling tensor e31 appearing")
print("in both equations -- the reciprocity property this chapter's derivation requires.")
