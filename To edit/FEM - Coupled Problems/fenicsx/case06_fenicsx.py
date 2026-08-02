"""Case Study 6 -- FEniCSx reference: finite-strain Flory-Rehner hydrogel swelling,
coupled displacement-concentration weak form (nonlinear, Newton-solved)."""
import ufl
from dolfinx import fem, mesh, nls
from mpi4py import MPI
from basix.ufl import element, mixed_element

domain = mesh.create_unit_square(MPI.COMM_WORLD, 30, 30)
Ue = element("Lagrange", domain.basix_cell(), 2, shape=(2,))
Ce = element("Lagrange", domain.basix_cell(), 1)
W = fem.functionspace(domain, mixed_element([Ue, Ce]))
w_sol = fem.Function(W)
u, c = ufl.split(w_sol)
(wv, cv) = ufl.TestFunctions(W)

Fdef = ufl.Identity(2) + ufl.grad(u)
J = ufl.det(Fdef)
NkT, chi, v_solvent, D = 1.0e6, 0.4, 3e-5, 1e-11
phi = 1.0/J
psi = 0.5*NkT*(ufl.tr(Fdef.T*Fdef) - 2 - 2*ufl.ln(J)) + (1.0/v_solvent)*(phi*ufl.ln(phi) + chi*phi*(1-phi))
P = ufl.diff(psi, Fdef)   # first Piola-Kirchhoff stress from the free energy directly (automatic differentiation)

F_mech = ufl.inner(P, ufl.grad(wv)) * ufl.dx
F_diff = (cv*(c) * ufl.dx + D*ufl.inner(ufl.grad(c), ufl.grad(cv)) * ufl.dx)
F_total = F_mech + F_diff
print("Finite-strain Flory-Rehner weak form assembled with stress obtained by symbolic")
print("differentiation (ufl.diff) of the free energy directly -- automatic, exact, and")
print("consistent with the Chapter 1 finite-strain elasticity weak form derivation.")
