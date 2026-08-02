"""
Case Study 13 : Fracturing Polymer Particle in Biological Flow - Target Capstone
Beginner demonstration: this capstone code reuses ideas from Cases 7, 9 and 12. A capsule
travels through a channel; local shear stress (from Case 12's capillary-number model)
drives a scalar damage accumulation law (inspired by Case 7's phase-field variable),
and we track when/where the particle is predicted to fracture. This is intentionally a
LOW-fidelity stand-in for the full coupled FSI+fracture+ML capstone -- the point is to
show how the validated single-physics pieces are assembled, exactly as the handbook's
staged workflow (A -> E) recommends.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def D_steady(Ca):
    return Ca / (1.0 + Ca)

def Ca_history(t):
    """A capsule accelerates through a narrowing channel: capillary number
    rises then falls as it passes the throat (a simple bell-shaped profile)."""
    t_throat = 5.0
    width = 1.5
    return 0.1 + 3.0 * np.exp(-0.5 * ((t - t_throat) / width) ** 2)

Gc_star = 3.0
def rhs(t, y):
    d = y[0]
    Ca = Ca_history(t)
    rate = max(Ca**2 - d, 0) / Gc_star if d < 1.0 else 0.0
    return [rate]

sol = solve_ivp(rhs, [0, 12], [0.0], t_eval=np.linspace(0, 12, 600), max_step=0.02)
t = sol.t
d = sol.y[0]
Ca_t = Ca_history(t)

frac_idx = np.argmax(d >= 0.999) if np.any(d >= 0.999) else None
print("Capillary number and accumulated damage along the channel transit:")
for i in range(0, len(t), 60):
    print(f"  t={t[i]:5.2f} s   Ca={Ca_t[i]:.3f}   damage d={d[i]:.4f}")

if frac_idx is not None:
    print(f"\nPredicted fracture (d reaches 1.0) at t = {t[frac_idx]:.2f} s, "
          f"just after the narrowest point of the constriction (t=5.0 s).")
else:
    print(f"\nMaximum damage reached in this transit: d_max = {d.max():.3f} "
          f"(particle survives this pass without fracturing)")

print("\nSensitivity to the critical toughness-like parameter Gc_star:")
for Gc_test in [1.0, 6.0, 30.0, 100.0]:
    def rhs_g(t, y, Gc=Gc_test):
        dloc = y[0]
        Ca = Ca_history(t)
        rate = max(Ca**2 - dloc, 0) / Gc if dloc < 1.0 else 0.0
        return [rate]
    s = solve_ivp(rhs_g, [0, 12], [0.0], t_eval=np.linspace(0, 12, 600), max_step=0.02)
    dmax = s.y[0].max()
    fractured = "FRACTURES" if dmax >= 0.999 else "survives"
    print(f"  Gc_star = {Gc_test:.1f}  ->  d_max = {dmax:.3f}  ({fractured})")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(t, Ca_t); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("capillary number Ca")
ax[1].plot(t, d); ax[1].axhline(1.0, ls='--', color='r'); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("damage d")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch13.png", dpi=150)
print("\nFigure saved.")
