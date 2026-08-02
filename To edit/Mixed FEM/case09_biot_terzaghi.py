"""
Capstone Project 9 (Case Study 9): Biot poroelasticity -- Terzaghi 1D
consolidation embedded in a 2D rectangle, genuine three-field
(displacement P2, Darcy flux RT0, pressure DG0) mixed formulation,
backward Euler in time, verified against the classical Terzaghi
analytical series solution.

Setup: soil column [0,1] x [0,H], drained at the top (y=H, p=0), zero
lateral displacement and no-flow on the sides and base (oedometer-cell
idealization), instantaneous unit traction applied at t=0+ at the top.

Reference:
  K. Terzaghi, "Erdbaumechanik auf bodenphysikalischer Grundlage,"
  Deuticke, Vienna, 1925.
  I. Ambartsumyan, E. Khattatov, I. Yotov, "A coupled multipoint
  stress-multipoint flux mixed finite element method for the Biot
  system of poroelasticity," Comput. Methods Appl. Mech. Engrg. 372,
  2020, 113407 (this capstone implements the simpler, compatible
  three-field RT0/P2/DG0 formulation rather than the multipoint
  MSMFE-MFMFE scheme of that paper; see Case Study 15 for an explicit
  discrepancy discussion of that distinction).
"""
import numpy as np
from skfem import *
from skfem.helpers import dot, ddot, sym_grad, div

E_mod, nu = 3.0e0, 0.2
mu_ = E_mod / (2 * (1 + nu))
lam_ = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
alpha_biot = 1.0
c0_storage = 1.0e-3
Kperm = 1.0e-2
H = 5.0
Wd = 1.0
p0_load = 1.0
dt = 2.0
n_steps = 60

mesh = MeshTri.init_tensor(np.linspace(0, Wd, 6), np.linspace(0, H, 31))

Uv = ElementVector(ElementTriP2())
RTf = ElementTriRT0()
Pp = ElementTriP0()
e = Uv * RTf * Pp
basis = Basis(mesh, e, intorder=4)


@BilinearForm
def a_transient(u, z, p, v, y, q, w):
    return (2 * mu_ * ddot(sym_grad(u), sym_grad(v)) + lam_ * div(u) * div(v)
            - alpha_biot * p * div(v)
            + (1.0 / Kperm) * dot(z, y) - p * div(y)
            - (alpha_biot / dt) * div(u) * q - (c0_storage / dt) * p * q
            - div(z) * q)


A = a_transient.assemble(basis)

top_facets = mesh.facets_satisfying(lambda x: x[1] > H - 1e-9)
bottom_facets = mesh.facets_satisfying(lambda x: x[1] < 1e-9)
left_facets = mesh.facets_satisfying(lambda x: x[0] < 1e-9)
right_facets = mesh.facets_satisfying(lambda x: x[0] > Wd - 1e-9)
side_bottom = np.concatenate([left_facets, right_facets, bottom_facets])

D_all = basis.get_dofs(side_bottom)


@LinearForm
def top_traction(v, y, q, w):
    return -p0_load * v[1] * (w.x[1] > H - 1e-9)


fb_top = FacetBasis(mesh, e, facets=top_facets)
F_load = top_traction.assemble(fb_top)

u_b, z_b, p_b = basis.split_bases()
n_u, n_z, n_p = u_b.N, z_b.N, p_b.N
offset_p = n_u + n_z


@BilinearForm
def div_u_q_form(u, q, w):
    return div(u) * q


@BilinearForm
def mass_pp_form(p, q, w):
    return p * q


M_up = div_u_q_form.assemble(u_b, p_b)   # shape (n_p, n_u)
M_pp = mass_pp_form.assemble(p_b)        # shape (n_p, n_p)


def terzaghi_pressure(zeta, Tv, n_terms=400):
    """Classical normalized Terzaghi excess pore pressure series;
    zeta measured from the drainage boundary (top), 0<=zeta<=1."""
    total = 0.0
    for m in range(n_terms):
        Mm = np.pi / 2 * (2 * m + 1)
        total += (2.0 / Mm) * np.sin(Mm * zeta) * np.exp(-Mm**2 * Tv)
    return total


cv = Kperm * (2 * mu_ + lam_)  # 1D consolidation coefficient, consistent units

x_prev = basis.zeros()
print(f"{'step':>5} {'time':>8} {'p(mid,FEM)':>12} {'p(mid,Terzaghi)':>16} "
      f"{'rel.diff':>10} {'mass balance residual':>22}")

top_facets_fb = FacetBasis(mesh, e, facets=top_facets)

for step in range(1, n_steps + 1):
    (u_prev, ub), (z_prev, zb), (p_prev, pb) = basis.split(x_prev)

    rhs = F_load.copy()
    rhs[offset_p:offset_p + n_p] += (alpha_biot / dt) * (M_up @ u_prev) \
        + (c0_storage / dt) * (M_pp @ p_prev)

    Acon, bcon, x0, I = condense(A, rhs, D=D_all)
    x_free = solve(Acon, bcon)
    x_new = np.zeros(A.shape[0])
    x_new[I] = x_free
    x_prev_for_balance = x_prev
    x_prev = x_new

    (u_h, ub), (z_h, zb), (p_h, pb) = basis.split(x_prev)
    (u_old, _), (z_old, _), (p_old, _) = basis.split(x_prev_for_balance)

    # global mass balance: change in stored fluid volume (compressibility
    # + volumetric strain) over this step should equal the net outflow
    # through the only drained (non-no-flow) boundary, the top
    storage_change = (c0_storage * (M_pp @ (p_h - p_old)).sum()
                       + alpha_biot * (M_up @ (u_h - u_old)).sum())

    @Functional
    def outflow_top(w):
        return dot(w["zh"], w.n)

    z_h_full = top_facets_fb.interpolate(x_prev)[1]
    net_outflow = outflow_top.assemble(top_facets_fb, zh=z_h_full) * dt
    mass_residual = storage_change + net_outflow  # should be ~0

    p_mid = pb.probes(np.array([[Wd / 2], [H / 2]])) @ p_h
    t_now = step * dt
    Tv = cv * t_now / H**2
    p_terzaghi = p0_load * terzaghi_pressure(0.5, Tv)
    rel = abs(p_mid[0] - p_terzaghi) / (abs(p_terzaghi) + 1e-9)
    if step % 5 == 0 or step == 1:
        print(f"{step:5d} {t_now:8.3f} {p_mid[0]:12.5f} {p_terzaghi:16.5f} "
              f"{rel:10.3f} {mass_residual:22.3e}")
