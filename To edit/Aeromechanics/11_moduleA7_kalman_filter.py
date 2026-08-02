"""
Capstone 11 -- Kalman Filter Observer for Unmeasured Structural States

Goal
----
A strain gauge on a blade root measures strain (proportional to
displacement/moment) but not velocity, and the signal is noisy. A Kalman
filter observer fuses the noisy strain measurement with a structural
model to reconstruct the *full* state (both displacement and velocity)
optimally -- exactly what is needed to feed a real-time flutter/vibration
monitoring or control system (see also Module A8) with variables it
cannot measure directly.

Method
------
1. SDOF mass-spring-damper truth model, simulated with process noise
   (unmodelled excitation) via a discretized state-space model.
2. Noisy displacement-only measurement (strain-gauge proxy).
3. Discrete-time Kalman filter: predict/update recursion using the same
   (known, from a prior test) structural model.
4. Compare the KF's estimated velocity (never directly measured) against
   the true simulated velocity, and quantify estimation error reduction
   vs. a naive finite-difference velocity estimate from the raw noisy
   measurement.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ---------------------------------------------------------------
# 1. SDOF truth system (blade-root-like local mode)
# ---------------------------------------------------------------
wn = 2 * np.pi * 15.0   # rad/s, 15 Hz local mode
zeta = 0.02
m = 1.0
k = m * wn ** 2
c = 2 * zeta * wn * m

fs = 500.0
dt = 1 / fs
T = 10.0
t = np.arange(0, T, dt)
n = len(t)

Ac = np.array([[0, 1], [-k / m, -c / m]])
Bc = np.array([[0], [1 / m]])
from scipy.linalg import expm
n_aug = np.zeros((3, 3))
n_aug[:2, :2] = Ac
n_aug[:2, 2:3] = Bc
Md = expm(n_aug * dt)
Ad = Md[:2, :2]
Bd = Md[:2, 2:3]

process_noise_std = 0.5  # N, unmodelled random forcing (gusts, unmeasured excitation)
meas_noise_std = 0.0008  # m, strain-gauge-equivalent displacement noise

x_true = np.zeros((2, n))
for i in range(1, n):
    w = rng.normal(0, process_noise_std)
    x_true[:, i] = Ad @ x_true[:, i - 1] + (Bd.flatten() * w)

y_meas = x_true[0] + rng.normal(0, meas_noise_std, n)  # displacement-only measurement

# ---------------------------------------------------------------
# 2. Discrete-time Kalman filter (known model, tuned Q/R)
# ---------------------------------------------------------------
Q = Bd @ Bd.T * process_noise_std ** 2  # process noise covariance (mapped through B)
R = np.array([[meas_noise_std ** 2]])
H = np.array([[1.0, 0.0]])

x_est = np.zeros((2, n))
P = np.eye(2) * 1e-6
x_hat = np.zeros(2)

for i in range(1, n):
    # predict
    x_pred = Ad @ x_hat
    P_pred = Ad @ P @ Ad.T + Q
    # update
    y_innov = y_meas[i] - H @ x_pred
    S = H @ P_pred @ H.T + R
    K_gain = P_pred @ H.T @ np.linalg.inv(S)
    x_hat = x_pred + (K_gain.flatten() * y_innov)
    P = (np.eye(2) - K_gain @ H) @ P_pred
    x_est[:, i] = x_hat

# ---------------------------------------------------------------
# 3. Naive baseline: finite-difference velocity from the raw noisy signal
# ---------------------------------------------------------------
v_fd = np.gradient(y_meas, dt)

rmse_kf_disp = np.sqrt(np.mean((x_est[0] - x_true[0]) ** 2))
rmse_kf_vel = np.sqrt(np.mean((x_est[1] - x_true[1]) ** 2))
rmse_fd_vel = np.sqrt(np.mean((v_fd - x_true[1]) ** 2))
rmse_raw_disp = np.sqrt(np.mean((y_meas - x_true[0]) ** 2))

print("=== Module A7: Kalman Filter Observer ===")
print(f"Raw measurement RMSE (displacement)      : {rmse_raw_disp:.5f} m")
print(f"KF-estimated displacement RMSE            : {rmse_kf_disp:.5f} m "
      f"({100*(1-rmse_kf_disp/rmse_raw_disp):.1f}% reduction vs. raw)")
print(f"Naive finite-difference velocity RMSE     : {rmse_fd_vel:.4f} m/s")
print(f"KF-estimated velocity RMSE (never measured): {rmse_kf_vel:.4f} m/s "
      f"({100*(1-rmse_kf_vel/rmse_fd_vel):.1f}% reduction vs. finite difference)")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
window = slice(0, 1000)

axes[0, 0].plot(t[window], y_meas[window], lw=0.5, color="gray", label="noisy measurement")
axes[0, 0].plot(t[window], x_true[0, window], color="black", lw=1.2, label="true displacement")
axes[0, 0].plot(t[window], x_est[0, window], color="firebrick", lw=1.0, ls="--", label="KF estimate")
axes[0, 0].set_title("Displacement: measured vs. true vs. KF estimate")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("displacement [m]")
axes[0, 0].legend(fontsize=7)

axes[0, 1].plot(t[window], x_true[1, window], color="black", lw=1.2, label="true velocity")
axes[0, 1].plot(t[window], v_fd[window], color="gray", lw=0.5, label="naive finite-difference")
axes[0, 1].plot(t[window], x_est[1, window], color="firebrick", lw=1.0, ls="--", label="KF estimate")
axes[0, 1].set_title("Velocity (never directly measured)")
axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("velocity [m/s]")
axes[0, 1].legend(fontsize=7)

axes[1, 0].bar(["raw meas.", "KF"], [rmse_raw_disp, rmse_kf_disp], color=["gray", "firebrick"])
axes[1, 0].set_title("Displacement RMSE"); axes[1, 0].set_ylabel("RMSE [m]")

axes[1, 1].bar(["finite diff.", "KF"], [rmse_fd_vel, rmse_kf_vel], color=["gray", "firebrick"])
axes[1, 1].set_title("Velocity RMSE"); axes[1, 1].set_ylabel("RMSE [m/s]")

plt.tight_layout()
plt.savefig("outputs/moduleA7_kalman_filter.png", dpi=150)
print("Saved outputs/moduleA7_kalman_filter.png")
