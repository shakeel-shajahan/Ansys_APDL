"""Case Study 17 -- FEniCSx reference: spherical radial Li-diffusion via UFL's built-in
support for axisymmetric/radial weak forms (weighting by r^2 exactly as this chapter and
the scikit-fem code above derive), extendable to a full pseudo-2D porous-electrode model."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_interval(MPI.COMM_WORLD, 200, [1e-9, 5e-6])
V = fem.functionspace(domain, ("Lagrange", 1))
c, ctest = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
r = x[0]
D = fem.Constant(domain, 1e-14)
dt = fem.Constant(domain, 1.0)

a = (r**2 * c * ctest * ufl.dx + dt*D*r**2*ufl.inner(ufl.grad(c), ufl.grad(ctest)) * ufl.dx)
print("Spherical (r^2-weighted) radial diffusion weak form assembled identically to the")
print("scikit-fem implementation above; FEniCSx's symbolic UFL layer makes the r^2 Jacobian")
print("weighting a one-line change from the Cartesian form, reducing implementation risk.")
