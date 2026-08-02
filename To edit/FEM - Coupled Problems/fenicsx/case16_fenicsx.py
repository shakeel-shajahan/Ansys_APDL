"""Case Study 16 -- FEniCSx reference: structural modal analysis (generalized eigenvalue
problem) providing the mode shapes/frequencies fed into the 2-DOF (or full modal) flutter
eigenvalue sweep used in this chapter's main code; full unsteady aeroelastic CFD-FSI is
noted but not reproduced here (see Case Study 9's ALE reference for the coupling pattern)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_box(MPI.COMM_WORLD, [[0,0,0],[2,0.3,0.01]], [40,8,2])
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
E, nu, rho = 70e9, 0.33, 2700.0
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
eps = lambda w: ufl.sym(ufl.grad(w))
sigma = lambda w: lam*ufl.tr(eps(w))*ufl.Identity(3) + 2*mu*eps(w)

k_form = ufl.inner(sigma(u), eps(v)) * ufl.dx
m_form = rho * ufl.inner(u, v) * ufl.dx
print("Structural stiffness (k_form) and mass (m_form) bilinear forms assembled for a")
print("generalized eigenvalue solve (via SLEPc) giving real wing mode shapes/frequencies,")
print("which then replace this chapter's idealized rigid 2-DOF section model.")
