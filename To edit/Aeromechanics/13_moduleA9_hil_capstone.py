"""
Capstone 13 -- Hardware-in-the-Loop Wind-Load Alleviation (Capstone of
Track A)

Goal
----
Bring together Modules A2 (aeroelastic model), A7 (state observer), and
A8 (real-time ROM) into a single closed-loop demonstration: a feedback
pitch controller that uses the ROM + Kalman-filtered state estimate to
actively damp gust-induced loads on the pitch-plunge section in
real time -- a simulated hardware-in-the-loop (HIL) load-alleviation
system, the natural capstone task after building the aeroelastic model,
the observer, and the real-time solver separately.

Method
------
1. 2-DOF pitch-plunge ROM (from Module A8) at a sub-flutter airspeed,
   driven by a simulated turbulent gust disturbance acting on the plunge
   DOF.
2. Only plunge displacement is "measured" (noisy sensor); a Kalman
   filter (Module A7 pattern, extended to 4 states) estimates the full
   state in real time.
3. A state-feedback controller (LQR gain, computed offline once) commands
   a control-surface pitch-moment input using the filtered state
   estimate, run inside the same 1 ms real-time loop as Module A8.
4. Compare peak/RMS plunge displacement and root bending-moment proxy
   with the controller ON vs. OFF -- the load-alleviation performance
   metric an HIL test campaign would report.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

rng = np.random.default_rng(9)

# ---------------------------------------------------------------
# 1. Pitch-plunge model (same as Modules A2/A8), sub-flutter airspeed
# ---------------------------------------------------------------
b = 0.5; a_h = -0.4; rho = 1.225
m = 25.0; I_alpha = 5.0; S_alpha = 0.6
Kh = 8.0e4; Ka = 1.2e4; Ch = 15.0; Ca = 3.0
U = 45.0  # m/s, sub-flutter but with reduced margin (motivates active control)

Ms = np.array([[m, S_alpha], [S_alpha, I_alpha]])
Cs = np.array([[Ch, 0], [0, Ca]])
Ks = np.array([[Kh, 0], [0, Ka]])

Ka_aero = np.array([[0, 2 * np.pi * rho * U ** 2 * b],
                     [0, 2 * np.pi * rho * U ** 2 * b ** 2 * (0.5 + a_h)]])
Ca_aero = np.array([[2 * np.pi * rho * U * b, 2 * np.pi * rho * U * b ** 2 * (0.5 - a_h)],
                     [-2 * np.pi * rho * U * b ** 2 * (0.5 + a_h),
                      2 * np.pi * rho * U * b ** 3 * (0.5 + a_h) * (0.5 - a_h)]])
C_tot = Cs + Ca_aero
K_tot = Ks + Ka_aero
Minv = np.linalg.inv(Ms)
Z = np.zeros((2, 2)); I2 = np.eye(2)
A = np.block([[Z, I2], [-Minv @ K_tot, -Minv @ C_tot]])
B = np.block([[np.zeros((2, 1))], [Minv @ np.array([[0.0], [1.0]])]])  # control pitch-moment actuator
Bw = np.block([[np.zeros((2, 1))], [Minv @ np.array([[1.0], [0.0]])]])  # gust force enters via plunge

# ---------------------------------------------------------------
# 2. LQR controller design (offline, once)
# ---------------------------------------------------------------
Q = np.diag([5e4, 1e5, 10.0, 10.0])   # penalize plunge/pitch displacement most
R = np.array([[1e-3]])
P_are = solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.inv(R) @ B.T @ P_are

# ---------------------------------------------------------------
# 3. Simulate closed-loop (controller ON) vs. open-loop (OFF), both with
#    the same Kalman-filtered state estimate feeding the controller
# ---------------------------------------------------------------
dt = 1e-3
T_sim = 6.0
n = int(T_sim / dt)
t = np.arange(n) * dt

# turbulent gust force disturbance (AR(1)-filtered white noise)
gust = np.zeros(n)
alpha_ar = 0.98
white = rng.normal(0, 1.0, n)
for i in range(1, n):
    gust[i] = alpha_ar * gust[i - 1] + np.sqrt(1 - alpha_ar ** 2) * white[i]
gust *= 400.0  # N, gust force magnitude

meas_noise_std = 5e-5  # m, plunge sensor noise
H_meas = np.array([[1.0, 0, 0, 0]])
Qkf = Bw @ Bw.T * (50.0 ** 2)  # process noise mapped through gust channel (approx.)
Rkf = np.array([[meas_noise_std ** 2]])


def simulate(control_on):
    x = np.zeros((4, n))
    x_hat = np.zeros((4, n))
    P = np.eye(4) * 1e-4
    u_hist = np.zeros(n)
    for i in range(1, n):
        u = -(K_lqr @ x_hat[:, i - 1])[0] if control_on else 0.0
        u_hist[i] = u
        xdot = A @ x[:, i - 1] + B.flatten() * u + Bw.flatten() * gust[i - 1]
        x[:, i] = x[:, i - 1] + dt * xdot  # simple forward-Euler, small dt is adequate here
        y = x[0, i] + rng.normal(0, meas_noise_std)

        # Kalman filter predict/update using the same control input applied
        x_pred = x_hat[:, i - 1] + dt * (A @ x_hat[:, i - 1] + B.flatten() * u)
        F_d = np.eye(4) + A * dt
        P_pred = F_d @ P @ F_d.T + Qkf
        innov = y - H_meas @ x_pred
        S = H_meas @ P_pred @ H_meas.T + Rkf
        Kg = P_pred @ H_meas.T @ np.linalg.inv(S)
        x_hat[:, i] = x_pred + (Kg.flatten() * innov)
        P = (np.eye(4) - Kg @ H_meas) @ P_pred
    return x, u_hist


x_off, _ = simulate(control_on=False)
x_on, u_on = simulate(control_on=True)

# root bending-moment proxy ~ proportional to pitch stiffness moment + plunge coupling
M_root_off = Ka * x_off[1] + Kh * 0.05 * x_off[0]
M_root_on = Ka * x_on[1] + Kh * 0.05 * x_on[0]

print("=== Module A9: HIL Wind-Load Alleviation Capstone ===")
print(f"Controller OFF: plunge RMS = {np.sqrt(np.mean(x_off[0]**2))*1000:.3f} mm, "
      f"peak = {np.max(np.abs(x_off[0]))*1000:.3f} mm, "
      f"root-moment-proxy RMS = {np.sqrt(np.mean(M_root_off**2)):.1f}")
print(f"Controller ON : plunge RMS = {np.sqrt(np.mean(x_on[0]**2))*1000:.3f} mm, "
      f"peak = {np.max(np.abs(x_on[0]))*1000:.3f} mm, "
      f"root-moment-proxy RMS = {np.sqrt(np.mean(M_root_on**2)):.1f}")
rms_reduction = 100 * (1 - np.sqrt(np.mean(x_on[0]**2)) / np.sqrt(np.mean(x_off[0]**2)))
moment_reduction = 100 * (1 - np.sqrt(np.mean(M_root_on**2)) / np.sqrt(np.mean(M_root_off**2)))
print(f"Plunge RMS reduction from active control: {rms_reduction:.1f}%")
print(f"Root-moment-proxy RMS reduction         : {moment_reduction:.1f}%")
print(f"Peak control input used: {np.max(np.abs(u_on)):.2f} N*m-equivalent")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
window = slice(0, 3000)

axes[0, 0].plot(t[window], x_off[0, window] * 1000, color="firebrick", label="controller OFF")
axes[0, 0].plot(t[window], x_on[0, window] * 1000, color="seagreen", label="controller ON")
axes[0, 0].set_title("Plunge displacement under gust loading")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("plunge [mm]")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(t[window], M_root_off[window], color="firebrick", label="controller OFF")
axes[0, 1].plot(t[window], M_root_on[window], color="seagreen", label="controller ON")
axes[0, 1].set_title("Root bending-moment proxy")
axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("moment proxy [a.u.]")
axes[0, 1].legend(fontsize=8)

axes[1, 0].bar(["OFF", "ON"],
               [np.sqrt(np.mean(x_off[0]**2))*1000, np.sqrt(np.mean(x_on[0]**2))*1000],
               color=["firebrick", "seagreen"])
axes[1, 0].set_title(f"Plunge RMS: {rms_reduction:.0f}% reduction")
axes[1, 0].set_ylabel("RMS plunge [mm]")

axes[1, 1].plot(t[window], u_on[window], color="darkorange")
axes[1, 1].set_title("Active control-moment command")
axes[1, 1].set_xlabel("time [s]"); axes[1, 1].set_ylabel("control input [N*m-equiv.]")

plt.tight_layout()
plt.savefig("outputs/moduleA9_hil_capstone.png", dpi=150)
print("Saved outputs/moduleA9_hil_capstone.png")
