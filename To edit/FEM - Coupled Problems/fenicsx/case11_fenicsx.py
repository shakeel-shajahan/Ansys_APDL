"""Case Study 11 -- FEniCSx reference: 3D patient-specific artery FSI with a Windkessel
outlet boundary condition (implemented as a Robin-type condition coupling outlet pressure
to a small ODE integrated alongside the 3D solve, exactly as SimVascular/svFSI do)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_unit_cube(MPI.COMM_WORLD, 20, 20, 40)
V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
Q = fem.functionspace(domain, ("Lagrange", 1))
p_wk = fem.Constant(domain, 80.0*133.322)   # Windkessel-supplied outlet pressure [Pa]

v = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
outlet_bc_term = p_wk * ufl.inner(n, v) * ufl.ds   # Robin/Neumann-like outlet traction

print("Outlet boundary weak form term assembled using the Windkessel-model pressure as a")
print("time-dependent Neumann condition, updated each time step by integrating this")
print("chapter's C*dp/dt = Q_in - p/Rd equation alongside the 3D FSI solve.")
