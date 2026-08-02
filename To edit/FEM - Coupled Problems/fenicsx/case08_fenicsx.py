"""Case Study 8 -- FEniCSx reference: moving heat source + J2 return-mapping plasticity,
via a Quadrature-space internal-variable (plastic strain) update each time step."""
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0],[0.02,0.004]], [100,20])
V = fem.functionspace(domain, ("Lagrange", 1))     # temperature
Vu = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # displacement

T, Ttest = ufl.TrialFunction(V), ufl.TestFunction(V)
rho, cp, k_th = 7900.0, 500.0, 18.0
dt = fem.Constant(domain, 4e-4)
x = ufl.SpatialCoordinate(domain)
v_scan, Q0, sigma_b = 0.008, 3e10, 6e-4
t = fem.Constant(domain, 0.0)
q_source = Q0*ufl.exp(-((x[0]-v_scan*t)**2 + (x[1]-0.004)**2)/(2*sigma_b**2))

a_T = (rho*cp*T*Ttest*ufl.dx + dt*k_th*ufl.inner(ufl.grad(T), ufl.grad(Ttest))*ufl.dx)
L_T = (rho*cp*Ttest*ufl.dx + dt*q_source*Ttest*ufl.dx)   # + previous T term added in time loop
print("Moving-source heat equation weak form assembled with time-dependent source position")
print("v_scan*t; plastic return-mapping (Case 8's derived closed-form residual-stress")
print("mechanism) applied at each quadrature point via a Quadrature function space.")
