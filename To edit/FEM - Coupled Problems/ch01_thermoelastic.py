"""
Case Study 1 : Thermoelastic Plate with a Hole
Beginner demonstration code (staggered thermo-elastic coupling)

Step 1: Solve 1-D transient heat conduction with an explicit finite-difference scheme.
Step 2: Transfer the resulting temperature field to a simple thermal-stress formula.
Step 3: Compare the stress concentration around a circular hole using the analytical
        Kirsch solution (this is the "reference" every FE result must match).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Step 1 : transient heat conduction (explicit FD) ----------
L = 0.2          # plate length in the x-direction [m]
nx = 41          # number of nodes
dx = L / (nx - 1)
alpha = 1.1e-5   # thermal diffusivity of steel [m^2/s]
dt = 0.4 * dx**2 / alpha        # stability limit: alpha*dt/dx^2 <= 0.5
n_steps = 400

T = np.zeros(nx)
T[0] = 80.0      # left face suddenly heated to 80 C
T[-1] = 20.0     # right face held at 20 C
T_hist = [T.copy()]

for step in range(n_steps):
    T_new = T.copy()
    for i in range(1, nx - 1):
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
    T_new[0] = 80.0
    T_new[-1] = 20.0
    T = T_new
    if step % 80 == 0:
        T_hist.append(T.copy())

print("Final steady-ish temperature profile (deg C), sampled every 10 nodes:")
print(np.round(T[::10], 2))

# ---------- Step 2 : one-way thermal stress in a fully restrained bar ----------
E = 200e9        # Young's modulus, steel [Pa]
alpha_T = 12e-6  # thermal expansion coefficient [1/K]
T0 = 20.0        # stress-free reference temperature [C]

sigma_thermal = -E * alpha_T * (T - T0)   # sigma = -E*alpha*(T-T0) for full restraint
print("\nThermal stress at each node if the bar is fully restrained (MPa):")
print(np.round(sigma_thermal[::10] / 1e6, 2))

# ---------- Step 3 : Kirsch analytical stress concentration around a hole ----------
def kirsch_sigma_theta(sigma_inf, a, r, theta):
    """Tangential stress around a circular hole in an infinite plate under
    uniaxial remote stress sigma_inf (Kirsch, 1898)."""
    ratio = a / r
    term1 = sigma_inf / 2 * (1 + ratio**2)
    term2 = sigma_inf / 2 * (1 + 3 * ratio**4) * np.cos(2 * theta)
    return term1 + term2

sigma_inf = 100e6   # remote applied stress [Pa]
a = 0.01            # hole radius [m]
theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
sigma_theta_at_edge = kirsch_sigma_theta(sigma_inf, a, a, theta)

print("\nKirsch tangential stress around the hole edge (MPa) at 8 angles:")
print(np.round(sigma_theta_at_edge / 1e6, 2))
print("Maximum stress concentration factor Kt = sigma_max / sigma_inf =",
      round(sigma_theta_at_edge.max() / sigma_inf, 3), " (classical value is 3.0)")

# ---------- plot ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
x = np.linspace(0, L, nx)
for k, Th in enumerate(T_hist):
    ax[0].plot(x, Th, label=f"step {k*80}")
ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("T [C]")
ax[0].set_title("Heat conduction, transient profiles")
ax[0].legend(fontsize=6)

theta_fine = np.linspace(0, 2*np.pi, 200)
sig_fine = kirsch_sigma_theta(sigma_inf, a, a, theta_fine) / 1e6
ax[1].plot(np.degrees(theta_fine), sig_fine)
ax[1].set_xlabel("angle theta [deg]"); ax[1].set_ylabel("sigma_theta [MPa]")
ax[1].set_title("Kirsch stress around hole edge")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch01.png", dpi=150)
print("\nFigure saved.")
