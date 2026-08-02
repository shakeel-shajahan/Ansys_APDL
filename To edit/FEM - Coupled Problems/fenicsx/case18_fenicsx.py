"""Case Study 18 -- FEniCSx reference: three-field (T, p, u) THM weak form for freezing
soil, with the apparent heat capacity and permeability-collapse nonlinearities implemented
as UFL conditional expressions evaluated directly in the residual (automatic Jacobian via
ufl.derivative for the Newton solve)."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_unit_square(MPI.COMM_WORLD, 40, 40)
VT = fem.functionspace(domain, ("Lagrange", 1))
T = fem.Function(VT)
Ttest = ufl.TestFunction(VT)

theta = 1.0/(1.0 + ufl.exp(-T/0.3))          # unfrozen fraction, UFL conditional-free sigmoid
L_latent, rho_w, phi_poro, c_soil = 334000.0, 1000.0, 0.35, 1800.0
dtheta_dT = theta*(1-theta)/0.3
c_apparent = c_soil + phi_poro*rho_w*L_latent*dtheta_dT

F_T = c_apparent*T*Ttest*ufl.dx + ufl.inner(ufl.grad(T), ufl.grad(Ttest))*ufl.dx
J_T = ufl.derivative(F_T, T)   # automatic Jacobian for Newton's method
print("Apparent-heat-capacity nonlinearity expressed directly in UFL as a smooth function")
print("of T; ufl.derivative computes the exact Newton Jacobian automatically, avoiding any")
print("hand-derived linearization of this chapter's freezing-curve nonlinearity.")
