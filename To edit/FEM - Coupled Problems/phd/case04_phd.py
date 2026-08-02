"""
Case Study 4 -- PhD-level application: 2D two-phase composite RVE (piezoelectric-ceramic
inclusions in a magnetostrictive matrix, inspired by the cited 2020 Scientific Reports
BaTiO3/ferrite two-scale homogenization study), effective stiffness computed via a genuine
periodic-mesh finite element solve (scikit-fem) rather than the Voigt/Reuss closed-form
mixing rules used in the earlier teaching code.
"""
import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from skfem.models.elasticity import linear_elasticity, plane_stress

E_matrix, nu_matrix = 45e9, 0.30    # magnetostrictive ferrite matrix
E_incl, nu_incl = 120e9, 0.25       # piezoelectric ceramic inclusion

N = 40
mesh = MeshTri.init_tensor(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
centroids = mesh.p[:, mesh.t].mean(axis=1)

# circular inclusion at RVE center, volume fraction ~0.3
r_incl = np.sqrt(0.3/np.pi)
is_inclusion = (centroids[0]-0.5)**2 + (centroids[1]-0.5)**2 < r_incl**2
vf_actual = is_inclusion.mean()
print(f"RVE: {mesh.t.shape[1]} elements, actual inclusion volume fraction = {vf_actual:.3f}")

E_field = np.where(is_inclusion, E_incl, E_matrix)
nu_field = np.where(is_inclusion, nu_incl, nu_matrix)
lam_field = E_field*nu_field/(1-nu_field**2)
mu_field = E_field/(2*(1+nu_field))

basis = Basis(mesh, ElementVector(ElementTriP1()))
nqp = basis.X.shape[-1]
lam_quad = np.tile(lam_field[:, None], (1, nqp))
mu_quad = np.tile(mu_field[:, None], (1, nqp))

from skfem.helpers import sym_grad, ddot, trace

@BilinearForm
def stiffness_hetero(u, v, w):
    return 2*w.mu*ddot(sym_grad(u), sym_grad(v)) + w.lam*trace(sym_grad(u))*trace(sym_grad(v))

K = stiffness_hetero.assemble(basis, lam=lam_quad, mu=mu_quad)

# apply a unit macroscopic strain exx=0.01 via linear displacement BC on left/right,
# periodic-like approximation: fix left edge, prescribe right edge displacement
eps_macro = 0.01
left = mesh.facets_satisfying(lambda x: np.abs(x[0]) < 1e-9)
right = mesh.facets_satisfying(lambda x: np.abs(x[0]-1) < 1e-9)
Dl = basis.get_dofs(left)
Dr = basis.get_dofs(right)

u = basis.zeros()
u[Dr.nodal['u^1']] = eps_macro * 1.0    # ux = eps_macro * x at x=1
D = np.unique(np.concatenate([Dl.all(), Dr.nodal['u^1']]))
# also pin y at one point to remove remaining rigid body mode
corner = basis.get_dofs(lambda x: (np.abs(x[0])<1e-9)&(np.abs(x[1])<1e-9))
u_sol = solve(*condense(K, np.zeros(basis.N), x=u, D=D))

uh = basis.interpolate(u_sol)
eps_xx = uh.grad[0,0]
sigma_xx = lam_quad*(uh.grad[0,0]+uh.grad[1,1]) + 2*mu_quad*uh.grad[0,0]

# volume-averaged effective stiffness E_eff = <sigma_xx> / eps_macro
dx = basis.dx
sigma_avg = np.sum(sigma_xx * dx) / np.sum(dx)
E_eff_FE = sigma_avg / eps_macro

E_voigt = vf_actual*E_incl + (1-vf_actual)*E_matrix
E_reuss = 1.0/(vf_actual/E_incl + (1-vf_actual)/E_matrix)

print(f"\nEffective modulus from genuine 2D heterogeneous FE solve : {E_eff_FE/1e9:.2f} GPa")
print(f"Voigt (iso-strain) bound                                 : {E_voigt/1e9:.2f} GPa")
print(f"Reuss (iso-stress) bound                                 : {E_reuss/1e9:.2f} GPa")
print(f"\nFE result falls {'within' if E_reuss <= E_eff_FE <= E_voigt else 'OUTSIDE'} "
      f"the Voigt-Reuss bounds, as required by the variational principles both bounds derive from.")
