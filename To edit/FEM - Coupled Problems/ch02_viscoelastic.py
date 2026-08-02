"""
Case Study 2 : Thermoviscoelastic Polymer under Cyclic Heating
Beginner demonstration: 3-branch generalized Maxwell model driven by a cyclic strain,
with viscous dissipation fed into a lumped heat balance (one-way mechanical -> thermal
coupling, the simplest form of the coupling chain described in the handbook).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- material parameters (generalized Maxwell, 2 branches + equilibrium spring) ----------
E_inf = 5e6        # long-term (equilibrium) modulus [Pa]
E1, tau1 = 20e6, 0.5      # branch 1 stiffness [Pa] and relaxation time [s]
E2, tau2 = 8e6, 5.0       # branch 2 stiffness [Pa] and relaxation time [s]

# ---------- cyclic loading ----------
freq = 0.5                 # Hz
omega = 2 * np.pi * freq
strain_amp = 0.02
t = np.linspace(0, 6, 3000)
dt = t[1] - t[0]
eps = strain_amp * np.sin(omega * t)
eps_dot = strain_amp * omega * np.cos(omega * t)

# ---------- integrate the internal (viscous) stresses q_i ----------
q1 = np.zeros_like(t)
q2 = np.zeros_like(t)
for i in range(1, len(t)):
    q1[i] = q1[i-1] + dt * (-q1[i-1] / tau1 + E1 * eps_dot[i-1])
    q2[i] = q2[i-1] + dt * (-q2[i-1] / tau2 + E2 * eps_dot[i-1])

sigma = E_inf * eps + q1 + q2

# ---------- dissipated power and lumped temperature rise ----------
# Dissipation rate per branch: q_i^2 / (E_i * tau_i)  (standard rheological identity)
diss_rate = q1**2 / (E1 * tau1) + q2**2 / (E2 * tau2)
rho, c = 1200.0, 1800.0     # density [kg/m^3], specific heat [J/kg K] for a generic polymer
beta = 1.0                  # fraction of dissipation converted to heat
T = np.zeros_like(t)
T[0] = 20.0
for i in range(1, len(t)):
    T[i] = T[i-1] + dt * beta * diss_rate[i-1] / (rho * c)

# ---------- report key numbers ----------
cycles_done = freq * t[-1]
print(f"Simulated {cycles_done:.1f} loading cycles at {freq} Hz, strain amplitude {strain_amp}")
print("Peak stress in the last cycle (MPa):", round(sigma[-200:].max() / 1e6, 3))
loop_energy = np.trapezoid(sigma[-int(1/freq/dt):], eps[-int(1/freq/dt):])
print("Hysteresis loop area of the last cycle (dissipated energy density, J/m^3):",
      round(loop_energy, 2))
print("Temperature rise after 6 s of cyclic loading (deg C):", round(T[-1] - T[0], 4))

# storage/loss modulus check at this frequency (closed form for Maxwell branches)
Estor = E_inf + E1 * (omega*tau1)**2/(1+(omega*tau1)**2) + E2 * (omega*tau2)**2/(1+(omega*tau2)**2)
Eloss = E1 * (omega*tau1)/(1+(omega*tau1)**2) + E2 * (omega*tau2)/(1+(omega*tau2)**2)
print(f"Analytical storage modulus E' = {Estor/1e6:.3f} MPa, loss modulus E'' = {Eloss/1e6:.3f} MPa")
print("tan(delta) = E''/E' =", round(Eloss/Estor, 4))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(t, eps); ax[0].set_title("Applied strain"); ax[0].set_xlabel("t [s]")
ax[1].plot(eps[-600:], sigma[-600:]/1e6); ax[1].set_title("Hysteresis loop (last cycles)")
ax[1].set_xlabel("strain"); ax[1].set_ylabel("stress [MPa]")
ax[2].plot(t, T); ax[2].set_title("Lumped temperature rise"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("T [C]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch02.png", dpi=150)
print("Figure saved.")
