"""
Case Study 10 : Vortex-Induced Vibration of an Elastic Cylinder
Beginner demonstration: the classical Facchinetti/wake-oscillator reduced-order model
couples a structural oscillator (cylinder cross-flow displacement y) to a Van der Pol
wake oscillator (q, representing the near-wake lift coefficient). This captures lock-in
without needing a full CFD solve, matching the handbook's "reduced-order vs full FEM"
comparison idea.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- structural and fluid parameters ----------
mass_ratio = 4.0       # m* = structural mass / displaced fluid mass
zeta = 0.01            # structural damping ratio
St = 0.2               # Strouhal number
A_wake = 12.0          # wake-oscillator nonlinearity coefficient (Facchinetti et al.)
eps_coupling = 0.3     # fluid-to-structure coupling coefficient
Cl0 = 0.3              # reference lift coefficient amplitude

def rhs(t, state, Ur):
    y, ydot, q, qdot = state
    omega_f = 2*np.pi*St*Ur           # vortex shedding frequency (reduced form)
    omega_s = 2*np.pi                 # structural natural frequency (normalized to 1 Hz)
    ydd = (-2*zeta*omega_s*ydot - omega_s**2*y + (Cl0/(0.5*mass_ratio))*q)
    qdd = omega_f**2*(A_wake*(1-q**2)*qdot/omega_f - q) + eps_coupling*ydd
    return [ydot, ydd, qdot, qdd]

Ur_list = np.linspace(3, 9, 25)     # reduced velocity sweep U/(f_n*D)
amp_list = []
for Ur in Ur_list:
    sol = solve_ivp(rhs, [0, 120], [0.01, 0, 0.1, 0], args=(Ur,),
                     t_eval=np.linspace(90, 120, 3000), max_step=0.01)
    y = sol.y[0]
    amp_list.append(y[len(y)//2:].max() - y[len(y)//2:].min())

amp_list = np.array(amp_list) / 2   # convert peak-to-peak to amplitude (diameters)

print("Reduced velocity sweep and resulting cross-flow amplitude (lock-in curve):")
for Ur, A in zip(Ur_list, amp_list):
    print(f"  Ur = {Ur:.2f}   A/D = {A:.4f}")

Ur_lockin_center = 1.0 / St
print(f"\nExpected lock-in center near Ur = 1/St = {Ur_lockin_center:.2f} "
      f"(observed peak at Ur = {Ur_list[np.argmax(amp_list)]:.2f})")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(Ur_list, amp_list, 'o-')
ax.axvline(Ur_lockin_center, ls='--', color='gray', label="1/St reference")
ax.set_xlabel("reduced velocity Ur = U/(fn D)"); ax.set_ylabel("A/D (cross-flow amplitude)")
ax.set_title("Wake-oscillator lock-in curve"); ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch10.png", dpi=150)
print("Figure saved.")
