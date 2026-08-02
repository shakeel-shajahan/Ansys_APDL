"""
Case Study 14 : Conjugate Heat Transfer in a Cooled Plate or Turbine Component
Beginner demonstration: two 1-D conduction domains (a hot solid plate and a cooling
fluid channel represented as advection-diffusion) are coupled at a shared interface by
iterating temperature and heat-flux continuity -- exactly the partitioned coupling
concept used by preCICE's heated-plate tutorial.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- solid domain (conduction only) ----------
Ls = 0.01          # solid thickness [m]
ks = 15.0          # solid thermal conductivity [W/mK]
ns = 21
xs = np.linspace(0, Ls, ns)
dxs = xs[1]-xs[0]
T_hot = 300.0      # fixed hot-side temperature [C]

# ---------- fluid domain (conduction/diffusion proxy for convection) ----------
Lf = 0.01
kf_eff = 40.0      # effective conductivity representing strong convective transport [W/mK]
nf = 21
xf = np.linspace(0, Lf, nf)
dxf = xf[1]-xf[0]
T_cold = 20.0      # fixed cold (far-field) fluid temperature [C]

Ts = np.linspace(T_hot, (T_hot+T_cold)/2, ns)   # initial guess
Tf = np.linspace((T_hot+T_cold)/2, T_cold, nf)

def solve_conduction_dirichlet(T, dx, k, T_left, T_right, n_iters=3000):
    """Simple steady-state 1D conduction relaxation with fixed Dirichlet ends."""
    T = T.copy()
    T[0] = T_left
    T[-1] = T_right
    for _ in range(n_iters):
        T[1:-1] = 0.5*(T[2:] + T[:-2])
    return T

print("Partitioned Dirichlet-Neumann iteration for the shared interface temperature:\n")
T_interface = (T_hot + T_cold) / 2
for outer in range(8):
    Ts = solve_conduction_dirichlet(Ts, dxs, ks, T_hot, T_interface)
    q_solid = -ks * (Ts[-1] - Ts[-2]) / dxs         # heat flux leaving the solid

    Tf = solve_conduction_dirichlet(Tf, dxf, kf_eff, T_interface, T_cold)
    q_fluid = -kf_eff * (Tf[1] - Tf[0]) / dxf       # heat flux entering the fluid

    flux_mismatch = q_solid - q_fluid
    # update the interface temperature to reduce the flux mismatch (simple relaxation)
    scale = ks / Ls + kf_eff / Lf     # exact sensitivity d(mismatch)/dT_interface for this linear problem
    T_interface = T_interface + 0.8 * flux_mismatch / scale   # slight under-relaxation (0.8) for a visible convergence sequence
    print(f"  iter {outer+1}: T_interface = {T_interface:6.2f} C   "
          f"q_solid = {q_solid:8.1f} W/m^2   q_fluid = {q_fluid:8.1f} W/m^2   "
          f"mismatch = {flux_mismatch:8.2f} W/m^2")

print(f"\nConverged interface temperature = {T_interface:.2f} C, "
      f"final flux mismatch = {flux_mismatch:.3f} W/m^2 (should approach 0)")

fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(xs*1e3, Ts, label="solid")
ax.plot((Lf-xf)*1e3 + Ls*1e3, Tf[::-1], label="fluid (mirrored)")
ax.set_xlabel("x [mm]"); ax.set_ylabel("T [C]"); ax.legend()
ax.set_title("Converged conjugate heat transfer profile")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch14.png", dpi=150)
print("Figure saved.")
