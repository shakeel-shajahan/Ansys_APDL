"""
Case Study 11 : Compliant Artery or Aneurysm FSI
Beginner demonstration: a 3-element Windkessel model represents the coupled effect of a
compliant vessel wall (capacitance C) on pulsatile flow, without needing a full 3D FSI
solve. This is the "0-D outlet boundary condition" used to close 3D FSI artery models.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- Windkessel parameters (illustrative, generic systemic circulation) ----------
Rp = 0.05     # proximal (characteristic) resistance [mmHg s/mL]
C  = 1.5      # arterial compliance [mL/mmHg]  (represents wall stiffness -> compliance link)
Rd = 1.0      # distal (peripheral) resistance [mmHg s/mL]

def Q_in(t, HR_bpm=70):
    """Idealized pulsatile cardiac inflow (half-sine systole + zero diastole)."""
    T = 60.0 / HR_bpm
    tt = t % T
    t_sys = 0.35 * T
    Q = np.where(tt < t_sys, 450 * np.sin(np.pi * tt / t_sys), 0.0)
    return Q

def rhs(t, y):
    P = y[0]
    Q = Q_in(t)
    dPdt = (Q - P / Rd) / C
    return [dPdt]

sol = solve_ivp(rhs, [0, 6], [80.0], max_step=0.001, t_eval=np.linspace(0, 6, 6000))
t = sol.t
P = sol.y[0]
Q = Q_in(t)

mask = t > 4   # look at a steady periodic cycle
print(f"Systolic pressure (steady cycle)  = {P[mask].max():.1f} mmHg")
print(f"Diastolic pressure (steady cycle) = {P[mask].min():.1f} mmHg")
print(f"Mean arterial pressure            = {P[mask].mean():.1f} mmHg")
print(f"Pulse pressure                    = {P[mask].max()-P[mask].min():.1f} mmHg")

# ---------- effect of wall stiffening (reduced compliance) on pulse pressure ----------
print("\nEffect of arterial stiffening (compliance C decreasing) on pulse pressure:")
for C_test in [2.5, 1.5, 0.8, 0.4]:
    def rhs_c(t, y, C=C_test):
        P = y[0]; Q = Q_in(t)
        return [(Q - P/Rd)/C]
    s = solve_ivp(rhs_c, [0, 6], [80.0], max_step=0.001, t_eval=np.linspace(4, 6, 2000))
    Pc = s.y[0]
    print(f"  C = {C_test:.2f} mL/mmHg  ->  pulse pressure = {Pc.max()-Pc.min():.1f} mmHg "
          f"(stiffer vessel = lower compliance = higher pulse pressure, matches clinical FSI observations)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(t[mask], Q[mask]); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("flow Q [mL/s]")
ax[1].plot(t[mask], P[mask]); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("pressure P [mmHg]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch11.png", dpi=150)
print("Figure saved.")
