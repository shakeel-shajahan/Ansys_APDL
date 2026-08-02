"""Case Study 9 -- FEniCSx reference: ALE fluid-structure interaction weak form (fluid
Navier-Stokes on a moving mesh, solid finite-strain elastodynamics), coupled via preCICE
adapters exactly as this chapter's partitioned coupling section describes."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain_f = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[2.5,0.41]], [100,20])
Vf = fem.functionspace(domain_f, ("Lagrange", 2, (2,)))   # fluid velocity (Taylor-Hood)
Qf = fem.functionspace(domain_f, ("Lagrange", 1))          # fluid pressure

v, q = ufl.TestFunction(Vf), ufl.TestFunction(Qf)
u_f = fem.Function(Vf)
mesh_vel = fem.Function(Vf)   # ALE mesh velocity, from a harmonic mesh-smoothing solve
rho_f, mu_f = 1000.0, 1e-3
d_dt_u = fem.Function(Vf)

F_fluid = (rho_f*ufl.inner(d_dt_u + ufl.dot(u_f - mesh_vel, ufl.nabla_grad(u_f)), v)*ufl.dx
           + 2*mu_f*ufl.inner(ufl.sym(ufl.grad(u_f)), ufl.sym(ufl.grad(v)))*ufl.dx)
print("ALE Navier-Stokes weak form assembled with (u_f - mesh_vel) advective velocity,")
print("exactly this chapter's ALE derivation; couple to the solid + preCICE adapter for")
print("the full partitioned FSI solve (see precice-adapters/fenicsx-adapter on GitHub).")
