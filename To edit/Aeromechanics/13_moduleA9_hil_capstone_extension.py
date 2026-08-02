"""
Extension to Capstone 13 -- Actuator-Saturation Robustness and LQR
Re-Tuning

Addresses reviewer feedback: "FPGA, Speedgoat, dSPACE, real sensors,
actuator latency, controller optimisation" -- of these, actuator
authority/saturation and controller re-tuning are the parts that can be
genuinely studied without real HIL hardware (which this sandbox does not
have); the rest (FPGA/Speedgoat/dSPACE) are explicitly out of scope, as
noted in the manual's introduction.

The base capstone's LQR controller assumed an ideal, unlimited actuator.
Real control-moment actuators (e.g. trailing-edge flaps, blade-pitch
actuators) have a hard authority limit. This extension:
  (1) imposes a realistic actuator saturation limit on the base
      capstone's controller and re-simulates the same gust disturbance,
      quantifying the resulting performance DEGRADATION relative to the
      unsaturated case reported in the base capstone;
  (2) re-tunes the LQR weighting (increasing control-effort penalty R)
      specifically to respect the saturation limit, and shows the
      resulting trade-off between load alleviation and control authority
      used -- directly answering the base capstone's own project-brief
      question about re-tuning under saturation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve_continuous_are

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])


def simulate_saturated(K_gain, u_limit, rng_local):
    x = np.zeros((4, n))
    x_hat = np.zeros((4, n))
    P = np.eye(4) * 1e-4
    u_hist = np.zeros(n)
    u_sat_hist = np.zeros(n)
    for i in range(1, n):
        u_cmd = -(K_gain @ x_hat[:, i - 1])[0]
        u_applied = np.clip(u_cmd, -u_limit, u_limit)
        u_hist[i] = u_cmd
        u_sat_hist[i] = u_applied
        xdot = A @ x[:, i - 1] + B.flatten() * u_applied + Bw.flatten() * gust[i - 1]
        x[:, i] = x[:, i - 1] + dt * xdot
        y = x[0, i] + rng_local.normal(0, meas_noise_std)

        x_pred = x_hat[:, i - 1] + dt * (A @ x_hat[:, i - 1] + B.flatten() * u_applied)
        F_d = np.eye(4) + A * dt
        P_pred = F_d @ P @ F_d.T + Qkf
        innov = y - H_meas @ x_pred
        S = H_meas @ P_pred @ H_meas.T + Rkf
        Kg = P_pred @ H_meas.T @ np.linalg.inv(S)
        x_hat[:, i] = x_pred + (Kg.flatten() * innov)
        P = (np.eye(4) - Kg @ H_meas) @ P_pred
    return x, u_hist, u_sat_hist


rng_ext = np.random.default_rng(1313)

# ---------------------------------------------------------------
# Part 1: impose a realistic actuator limit on the BASE capstone's
# already-tuned LQR gain, and measure the resulting degradation
# ---------------------------------------------------------------
u_limit_realistic = 25.0  # N*m-equivalent, a plausible actuator authority limit
                            # (base capstone's peak commanded input, unsaturated,
                            # was ~60 N*m-equivalent -- so this limit WILL bind)

x_sat, u_cmd_sat, u_applied_sat = simulate_saturated(K_lqr, u_limit_realistic, rng_ext)
frac_saturated = np.mean(np.abs(u_cmd_sat) > u_limit_realistic)

rms_plunge_sat = np.sqrt(np.mean(x_sat[0] ** 2)) * 1000
rms_plunge_unsat_base = np.sqrt(np.mean(x_on[0] ** 2)) * 1000  # from base capstone's own run
rms_plunge_off_base = np.sqrt(np.mean(x_off[0] ** 2)) * 1000

print("=== Extension 13: Actuator Saturation and LQR Re-Tuning ===")
print(f"Actuator limit imposed: +/-{u_limit_realistic:.0f} N*m-equivalent "
      f"(base capstone's unsaturated peak command was ~60 N*m-equivalent)")
print(f"Fraction of time the commanded input exceeds this limit (i.e. actually saturates): "
      f"{frac_saturated*100:.1f}%")
print(f"\nControl OFF (no controller)                    : plunge RMS = {rms_plunge_off_base:.3f} mm")
print(f"Base capstone's LQR, UNSATURATED (idealised)   : plunge RMS = {rms_plunge_unsat_base:.3f} mm "
      f"({100*(1-rms_plunge_unsat_base/rms_plunge_off_base):.1f}% reduction)")
print(f"SAME LQR gain, WITH realistic actuator saturation: plunge RMS = {rms_plunge_sat:.3f} mm "
      f"({100*(1-rms_plunge_sat/rms_plunge_off_base):.1f}% reduction)")
print(f"Saturation erodes {100*(rms_plunge_sat-rms_plunge_unsat_base)/(rms_plunge_off_base-rms_plunge_unsat_base):.1f}% "
      f"of the load-alleviation benefit the base capstone reported under the idealised (unsaturated) assumption.")

# ---------------------------------------------------------------
# Part 2: re-tune the LQR (increase control-effort penalty R) so the
# gain itself demands less authority, trading off alleviation
# performance for reduced saturation
# ---------------------------------------------------------------
R_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0]  # 1e-3 matches the base capstone's original tuning
results_retune = []
for R_val in R_values:
    Q_lqr = np.diag([5e4, 1e5, 10.0, 10.0])
    P_are = solve_continuous_are(A, B, Q_lqr, np.array([[R_val]]))
    K_r = np.linalg.inv(np.array([[R_val]])) @ B.T @ P_are
    x_r, u_cmd_r, u_applied_r = simulate_saturated(K_r, u_limit_realistic, np.random.default_rng(1313))
    rms_r = np.sqrt(np.mean(x_r[0] ** 2)) * 1000
    frac_sat_r = np.mean(np.abs(u_cmd_r) > u_limit_realistic)
    peak_cmd_r = np.max(np.abs(u_cmd_r))
    results_retune.append((R_val, rms_r, frac_sat_r, peak_cmd_r))
    print(f"R={R_val:8.4f}: plunge RMS = {rms_r:6.3f} mm, saturated {frac_sat_r*100:5.1f}% of the time, "
          f"peak command = {peak_cmd_r:6.1f} N*m-equiv.")

best_no_sat = min([r for r in results_retune if r[2] < 0.01], key=lambda z: z[1], default=None)
if best_no_sat:
    print(f"\nBest re-tuned gain that never saturates: R={best_no_sat[0]}, "
          f"plunge RMS = {best_no_sat[1]:.3f} mm "
          f"({100*(1-best_no_sat[1]/rms_plunge_off_base):.1f}% reduction vs. OFF, "
          f"vs. {100*(1-rms_plunge_unsat_base/rms_plunge_off_base):.1f}% for the original idealised design)")
else:
    print("\nNo tested R value fully avoids saturation -- the actuator limit is the binding constraint")
    print("regardless of LQR tuning at this gust intensity; a real design would need either a higher-")
    print("authority actuator or acceptance of some saturation-induced performance loss.")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
window = slice(0, 3000)

axes[0, 0].plot(t[window], x_off[0, window] * 1000, color="gray", label="controller OFF", lw=0.8)
axes[0, 0].plot(t[window], x_on[0, window] * 1000, color="seagreen", label="LQR, unsaturated (base)", lw=0.8)
axes[0, 0].plot(t[window], x_sat[0, window] * 1000, color="firebrick", label="LQR, WITH saturation", lw=0.8)
axes[0, 0].set_title("Plunge displacement: effect of actuator saturation")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("plunge [mm]")
axes[0, 0].legend(fontsize=7)

axes[0, 1].plot(t[window], u_cmd_sat[window], color="firebrick", label="commanded", lw=0.7)
axes[0, 1].plot(t[window], u_applied_sat[window], color="black", label="applied (saturated)", lw=0.7)
axes[0, 1].axhline(u_limit_realistic, color="k", ls="--", lw=0.7)
axes[0, 1].axhline(-u_limit_realistic, color="k", ls="--", lw=0.7)
axes[0, 1].set_title("Commanded vs. saturated control input")
axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("control input [N*m-equiv.]")
axes[0, 1].legend(fontsize=7)

Rs = [r[0] for r in results_retune]
rms_list = [r[1] for r in results_retune]
frac_list = [r[2] * 100 for r in results_retune]
ax2 = axes[1, 0]
ax2.semilogx(Rs, rms_list, "o-", color="steelblue")
ax2.set_xlabel("LQR control-effort weight R"); ax2.set_ylabel("plunge RMS [mm]", color="steelblue")
ax2.set_title("Re-tuning trade-off: performance vs. saturation")
ax2b = ax2.twinx()
ax2b.semilogx(Rs, frac_list, "s--", color="firebrick")
ax2b.set_ylabel("% time saturated", color="firebrick")

axes[1, 1].bar(["OFF", "LQR\n(unsaturated)", "LQR\n(saturated)"],
               [rms_plunge_off_base, rms_plunge_unsat_base, rms_plunge_sat],
               color=["gray", "seagreen", "firebrick"])
axes[1, 1].set_title("Summary: plunge RMS across scenarios")
axes[1, 1].set_ylabel("plunge RMS [mm]")

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA9_extension_actuator_saturation.png"), dpi=150)
print("Saved outputs/moduleA9_extension_actuator_saturation.png")
