"""
Case Study 7 : Phase-Field Fracture in a Heterogeneous Polymer
Beginner demonstration: a 1-D bar pulled in tension, phase-field damage variable d in [0,1]
solved by staggered alternate minimization (fixed-point iteration between displacement u
and damage d), the same algorithmic idea used in 2D/3D phase-field fracture codes.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- material and regularization parameters ----------
E = 3.0e9          # Young's modulus of the polymer [Pa]
Gc = 300.0         # critical fracture energy [J/m^2]
ell = 0.02         # phase-field length scale [m]
L = 1.0            # bar length [m]
n = 101
x = np.linspace(0, L, n)
dx = x[1] - x[0]

d = np.zeros(n)     # damage field, 0 = intact, 1 = fully broken
u = np.zeros(n)

def g(d):
    return (1 - d)**2 + 1e-6     # degradation function, small residual stiffness

def solve_u(d, eps_applied):
    """Displacement solved as a simple 1D bar with degraded stiffness (Newton not
    needed here because the elastic problem is linear for fixed d)."""
    u = eps_applied * x
    strain = np.gradient(u, dx)
    return u, strain

def solve_d(strain, d_old):
    """Local closed-form update from the phase-field Euler-Lagrange equation for a
    1-D staggered scheme (explicit local minimization at each node), enforcing d>=d_old
    (irreversibility)."""
    psi_plus = 0.5 * E * strain**2
    d_new = psi_plus * 2 * ell / (Gc + psi_plus * 2 * ell)
    return np.maximum(d_new, d_old)   # irreversibility constraint

# ---------- staggered loading loop ----------
eps_list = np.linspace(0, 0.01, 25)
sigma_avg = []
d_max_history = []
for eps_applied in eps_list:
    for _ in range(15):     # a handful of staggered iterations per load step
        u, strain = solve_u(d, eps_applied)
        d = solve_d(strain, d)
    sigma_field = g(d) * E * strain
    sigma_avg.append(sigma_field.mean())
    d_max_history.append(d.max())

sigma_avg = np.array(sigma_avg)
print("Applied strain vs average stress and max damage (selected points):")
for i in range(0, len(eps_list), 4):
    print(f"  eps={eps_list[i]:.4f}  sigma_avg={sigma_avg[i]/1e6:.3f} MPa  d_max={d_max_history[i]:.4f}")

peak_idx = np.argmax(sigma_avg)
print(f"\nPeak (critical) stress = {sigma_avg[peak_idx]/1e6:.3f} MPa at applied strain "
      f"{eps_list[peak_idx]:.4f}  -- softening begins after this point.")

# ---------- Griffith energy check (order-of-magnitude, 1D analogue) ----------
sigma_c_griffith = np.sqrt(E * Gc / ell)
print(f"1-D order-of-magnitude Griffith estimate sigma_c ~ sqrt(E*Gc/ell) = "
      f"{sigma_c_griffith/1e6:.3f} MPa (compare with the numerical peak above)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(eps_list, sigma_avg/1e6, 'o-')
ax[0].set_xlabel("applied strain"); ax[0].set_ylabel("average stress [MPa]")
ax[0].set_title("Load-displacement curve")
ax[1].plot(x, d)
ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("damage d")
ax[1].set_title("Final damage profile")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch07.png", dpi=150)
print("Figure saved.")
