"""
Case Study 8 : Thermo-Mechanical Fracture and Residual Stress in Additive Manufacturing
Beginner demonstration: a 1-D moving Gaussian heat source scans across a bar (representing
a single weld/print track). We solve transient conduction, then estimate residual stress
from the peak-temperature history using a simplified elastic-perfectly-plastic model.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- moving heat source ----------
L = 0.05          # track length [m]
nx = 101
x = np.linspace(0, L, nx)
dx = x[1] - x[0]
alpha = 5.5e-6    # thermal diffusivity, generic metal alloy [m^2/s]
k_th = 15.0       # thermal conductivity [W/mK]
rho, c = 7800.0, 500.0
v_scan = 0.01     # scan speed [m/s]
Q0 = 6.0e9        # source intensity [W/m^3], lumped 1D approximation
sigma_beam = 0.002  # beam width [m]

dt = 0.3 * dx**2 / alpha
t_total = L / v_scan * 1.4
n_steps = int(t_total / dt)

T = np.full(nx, 20.0)
T_peak = T.copy()

for step in range(n_steps):
    t = step * dt
    x_source = v_scan * t
    q = Q0 * np.exp(-(x - x_source)**2 / (2 * sigma_beam**2))
    T_new = T.copy()
    for i in range(1, nx - 1):
        cond = alpha * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])
        source_term = q[i] * dt / (rho * c)
        T_new[i] = T[i] + cond + source_term
    T_new[0], T_new[-1] = T_new[1], T_new[-2]   # insulated ends
    T = T_new
    T_peak = np.maximum(T_peak, T)

print("Peak temperature reached at each 10th node along the track (deg C):")
print(np.round(T_peak[::10], 1))
print(f"\nMaximum peak temperature = {T_peak.max():.1f} C  "
      f"(above ~1450 C would indicate local melting for this generic alloy)")

# ---------- simplified residual stress estimate (elastic-perfectly-plastic, 1D bar analogy) ----------
E = 200e9
alpha_T = 12e-6
sigma_y = 300e6
T0 = 20.0

# thermal strain if fully restrained
# thermal strain if fully restrained (relative to the reference/room temperature)
eps_thermal_peak = alpha_T * (T_peak - T0)
sigma_trial_heating = E * eps_thermal_peak   # magnitude of the compressive trial stress at peak T

# During heating the bar tries to expand but is restrained -> compressive stress builds up.
# Once |stress| = sigma_y the material yields plastically (compressive plastic strain grows,
# holding the stress at -sigma_y for the rest of the heating phase). On cooling back to T0 the
# thermal strain vanishes but the plastic strain remains, which is revealed as a TENSILE
# residual stress: sigma_residual = E*alpha*(Tmax-T0) - sigma_y, capped at +/- sigma_y because
# a large enough excursion can also yield the material in tension during cooling.
yielded = sigma_trial_heating > sigma_y
sigma_residual_raw = np.where(yielded, sigma_trial_heating - sigma_y, 0.0)
sigma_residual = np.clip(sigma_residual_raw, -sigma_y, sigma_y)

print("\nResidual stress after full cool-down, sampled every 10 nodes (MPa):")
print(np.round(sigma_residual[::10] / 1e6, 2))
print(f"Maximum residual (tensile) stress = {sigma_residual.max()/1e6:.1f} MPa "
      f"(yield strength is {sigma_y/1e6:.0f} MPa)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(x*1e3, T_peak); ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("peak T [C]")
ax[1].plot(x*1e3, sigma_residual/1e6); ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("residual stress [MPa]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch08.png", dpi=150)
print("Figure saved.")
