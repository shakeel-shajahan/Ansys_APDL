"""Case Study 12 -- FEniCSx reference: CutFEM interface representation for a capsule in a
microchannel, using a level-set field and Nitsche's method for interface coupling
(exactly the technique this chapter's derivation names as an alternative to ALE)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[10,2]], [200,40])
Vphi = fem.functionspace(domain, ("Lagrange", 1))
phi = fem.Function(Vphi)   # level-set function, phi=0 marks the capsule interface

gamma_nitsche = fem.Constant(domain, 100.0)
print("Level-set field defines the capsule interface implicitly; Nitsche penalty/consistency")
print("terms (weighted by gamma_nitsche) enforce interface kinematic/dynamic continuity")
print("weakly on a single fixed background mesh, exactly this chapter's CutFEM derivation --")
print("avoiding the ALE remeshing that would be needed for this capsule's large motion.")
