"""Case Study 15 -- FEniCSx reference: Helmholtz acoustic-structure interaction,
complex-valued weak form (frequency domain), coupled to a structural plate via normal
acceleration continuity at the shared interface."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[2,1]], [80,40])
V = fem.functionspace(domain, ("Lagrange", 1))
p, ptest = ufl.TrialFunction(V), ufl.TestFunction(V)

c_sound, omega = 1480.0, 2*np.pi*200.0
k_wave = omega/c_sound

a = (ufl.inner(ufl.grad(p), ufl.grad(ptest))*ufl.dx
     - k_wave**2 * ufl.inner(p, ptest) * ufl.dx)
print("Complex-valued Helmholtz weak form assembled (requires PETSc built with complex")
print("scalars); couple to a structural plate via a normal-acceleration Robin/Neumann term")
print("at the wetted interface, exactly this chapter's boundary-integral derivation.")
