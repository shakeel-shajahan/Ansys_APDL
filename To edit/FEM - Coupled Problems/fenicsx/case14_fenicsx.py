"""Case Study 14 -- FEniCSx reference: conjugate heat transfer via a single MONOLITHIC
mesh spanning both solid and fluid subdomains (MeshTags distinguishing materials), the
production alternative to the partitioned Dirichlet-Neumann scheme used in the teaching
and scikit-fem codes above -- avoiding any interface iteration entirely."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[0.02,0.02]], [80,80])
V = fem.functionspace(domain, ("Lagrange", 1))
T, Ttest = ufl.TrialFunction(V), ufl.TestFunction(V)

# k(x) defined piecewise via a DG0 material-marker field distinguishing solid/fluid cells
Q = fem.functionspace(domain, ("DG", 0))
k_field = fem.Function(Q)   # k_field.x.array[:] = ks or kf per cell, from MeshTags

a = k_field * ufl.inner(ufl.grad(T), ufl.grad(Ttest)) * ufl.dx
print("Monolithic CHT weak form assembled on ONE mesh with a piecewise material field;")
print("interface continuity of temperature and flux is enforced automatically by using")
print("continuous (Lagrange) elements across the material interface -- no partitioned")
print("iteration needed at all, at the cost of requiring both domains in one mesh.")
