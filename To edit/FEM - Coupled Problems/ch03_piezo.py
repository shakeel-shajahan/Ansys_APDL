"""
Case Study 3 : Piezoelectric Cantilever Energy Harvester
Beginner demonstration: single-degree-of-freedom electromechanical harvester model
(Erturk-Inman reduced form). Two-way coupling: mechanical motion generates voltage,
voltage feeds back an electrical damping force on the beam.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- lumped electromechanical parameters (illustrative, not a specific product) ----------
m = 0.012          # modal mass [kg]
k = 1800.0         # modal stiffness [N/m]
c = 0.25           # mechanical damping [N s/m]
theta = 8.5e-4     # electromechanical coupling [N/V] = [C/m]
Cp = 32e-9         # piezo capacitance [F]
R = 1.0e5          # load resistance [Ohm]

wn = np.sqrt(k/m)
print(f"Short-circuit natural frequency = {wn/2/np.pi:.2f} Hz")

F0 = 0.6           # base excitation force amplitude [N]
omega = wn         # drive at resonance for maximum harvested power

def rhs(t, y):
    x, v, V = y
    dx = v
    dv = (F0*np.cos(omega*t) - c*v - k*x - theta*V) / m
    dV = (theta*v - V/R) / Cp
    return [dx, dv, dV]

from scipy.integrate import solve_ivp
t_span = (0, 2.0)
t_eval = np.linspace(*t_span, 4000)
sol = solve_ivp(rhs, t_span, [0, 0, 0], t_eval=t_eval, max_step=1e-4)

x, v, V = sol.y
P_inst = V**2 / R
# use the last 1/4 second (steady state) to report power
mask = t_eval > 1.75
P_avg = P_inst[mask].mean()
V_rms = np.sqrt((V[mask]**2).mean())
x_amp = np.abs(x[mask]).max()

print(f"Steady-state tip displacement amplitude = {x_amp*1e3:.3f} mm")
print(f"Steady-state RMS voltage = {V_rms:.3f} V")
print(f"Average harvested power at R = {R:.0f} ohm : {P_avg*1e6:.3f} microW")

# open circuit vs short circuit frequency shift (checks 2-way coupling changes stiffness)
k_eff_oc = k + theta**2 / Cp     # open circuit adds an electrical stiffness term
wn_oc = np.sqrt(k_eff_oc/m)
print(f"Open-circuit natural frequency = {wn_oc/2/np.pi:.2f} Hz "
      f"(shift of {(wn_oc-wn)/2/np.pi:.3f} Hz confirms two-way coupling)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(t_eval, x*1e3); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("tip disp [mm]")
ax[1].plot(t_eval, V); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("Voltage [V]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch03.png", dpi=150)
print("Figure saved.")
