"""
Case Study 12 : Deformable Polymer Capsule or Microgel in a Microchannel
Beginner demonstration: a simplified small-deformation relaxation model (in the spirit of
Maffettone-Minale / Cox small-deformation theory) tracks a capsule's Taylor deformation
index D and inclination angle phi as it relaxes toward a capillary-number-dependent
steady shape under imposed shear flow -- illustrating the fluid-to-structure coupling
without needing a full 3D two-way FSI solve.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

tau_D = 1.0     # shape relaxation time (elastic capsule "memory")
tau_phi = 1.0   # orientation relaxation time

def D_steady(Ca):
    """Small-deformation-theory-like saturating steady deformation index."""
    return Ca / (1.0 + Ca)

def phi_steady(Ca):
    """Inclination angle rotates from 45 deg (elastic/weak-flow limit) toward 0 deg
    (flow-alignment-dominated limit) as the capillary number Ca increases."""
    return 45.0 / (1.0 + Ca)

def rhs(t, y, Ca):
    D, phi = y
    dDdt = (D_steady(Ca) - D) / tau_D
    dphidt = (phi_steady(Ca) - phi) / tau_phi
    return [dDdt, dphidt]

print("Steady capsule shape as a function of the capillary number Ca "
      "(ratio of viscous shear stress to elastic restoring stress):\n")
results = {}
for Ca in [0.05, 0.2, 0.5, 1.0, 2.0]:
    sol = solve_ivp(rhs, [0, 10], [0.0, 45.0], args=(Ca,),
                     t_eval=np.linspace(0, 10, 1000))
    D_final = sol.y[0][-1]
    phi_final = sol.y[1][-1]
    results[Ca] = sol
    print(f"  Ca = {Ca:.2f}  ->  steady D = {D_final:.4f}  (theory: {D_steady(Ca):.4f}),  "
          f"steady phi = {phi_final:.1f} deg  (theory: {phi_steady(Ca):.1f} deg)")

print("\nPhysical trend: D increases monotonically with Ca (softer capsule / stronger flow")
print("deforms more) while phi decreases from 45 deg toward 0 deg (the capsule progressively")
print("aligns with the flow direction as the flow stress dominates over elastic recovery).")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for Ca, sol in results.items():
    ax[0].plot(sol.t, sol.y[0], label=f"Ca={Ca}")
    ax[1].plot(sol.t, sol.y[1], label=f"Ca={Ca}")
ax[0].set_xlabel("time (strain units)"); ax[0].set_ylabel("deformation index D"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("time (strain units)"); ax[1].set_ylabel("inclination angle [deg]"); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch12.png", dpi=150)
print("Figure saved.")
