"""
Capstone 12 -- A Real-Time Reduced-Order Model (ROM) for the Pitch-Plunge
Aeroelastic System

Goal
----
Hardware-in-the-loop (HIL) rigs and real-time flutter-margin monitors
must integrate the aeroelastic equations of motion within a hard
real-time budget (e.g. every 1 ms). The full 2-DOF pitch-plunge model
from Module A2 is cheap enough on its own, but this capstone builds a
*reduced-order* modal/state-space formulation, benchmarks its per-step
wall-clock cost against a naive dense-matrix formulation, and verifies
that the ROM reproduces the full model's response to within a tight
tolerance -- exactly the verification step required before trusting a
ROM inside a real-time control loop.

Method
------
1. Reuse the 2-DOF pitch-plunge state matrix from Module A2 at a fixed
   sub-flutter airspeed.
2. "Full" model: 4x4 dense state matrix, explicit RK4 time integration.
3. ROM: project onto the 2 dominant complex-mode subspace (here, exact
   for a 4-state system, but the workflow generalises to large FE models
   where this is genuinely a reduction), producing a smaller effective
   integration cost via pre-factored modal exponential update.
4. Benchmark wall-clock time per step for both, over a fixed-step
   real-time loop, and verify the ROM tracks the full solution.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------
# 1. Reuse 2-DOF pitch-plunge structural + quasi-steady aero matrices
#    (same physical parameters as Module A2) at a fixed sub-flutter speed
# ---------------------------------------------------------------
b = 0.5; a_h = -0.4; rho = 1.225
m = 25.0; I_alpha = 5.0; S_alpha = 0.6
Kh = 8.0e4; Ka = 1.2e4; Ch = 15.0; Ca = 3.0

Ms = np.array([[m, S_alpha], [S_alpha, I_alpha]])
Cs = np.array([[Ch, 0], [0, Ca]])
Ks = np.array([[Kh, 0], [0, Ka]])

U = 40.0  # m/s, well below the ~63 m/s flutter speed found in Module A2


def aero_matrices(U):
    Ka_aero = np.array([[0, 2 * np.pi * rho * U ** 2 * b],
                         [0, 2 * np.pi * rho * U ** 2 * b ** 2 * (0.5 + a_h)]])
    Ca_aero = np.array([[2 * np.pi * rho * U * b, 2 * np.pi * rho * U * b ** 2 * (0.5 - a_h)],
                         [-2 * np.pi * rho * U * b ** 2 * (0.5 + a_h),
                          2 * np.pi * rho * U * b ** 3 * (0.5 + a_h) * (0.5 - a_h)]])
    return Ca_aero, Ka_aero


Ca_aero, Ka_aero = aero_matrices(U)
C_tot = Cs + Ca_aero
K_tot = Ks + Ka_aero
Minv = np.linalg.inv(Ms)
Z = np.zeros((2, 2)); I2 = np.eye(2)
A_full = np.block([[Z, I2], [-Minv @ K_tot, -Minv @ C_tot]])

# ---------------------------------------------------------------
# 2. Modal reduction: eigendecompose A_full, keep all modes here (n=4 is
#    already small) but structure the workflow exactly as one would for
#    a large FE model where only the first few aeroelastic modes matter
# ---------------------------------------------------------------
eigvals, eigvecs = np.linalg.eig(A_full)
Phi = eigvecs
Phi_inv = np.linalg.inv(Phi)
# ROM propagation over one fixed timestep dt: exact modal exponential update
dt = 1e-3  # 1 ms real-time step, typical HIL rate
Lambda_d = np.exp(eigvals * dt)  # diagonal discrete modal propagator

# ---------------------------------------------------------------
# 3. Simulate both formulations over a real-time loop with an initial
#    pitch disturbance, and benchmark per-step wall-clock cost
# ---------------------------------------------------------------
T_sim = 3.0
n_steps = int(T_sim / dt)
x0 = np.array([0.0, 0.05, 0.0, 0.0])  # small initial pitch angle [rad]

# --- Full dense-matrix RK4 integration ---
def rk4_step(x, A, dt):
    k1 = A @ x
    k2 = A @ (x + 0.5 * dt * k1)
    k3 = A @ (x + 0.5 * dt * k2)
    k4 = A @ (x + dt * k3)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


x_full = np.zeros((4, n_steps))
x_full[:, 0] = x0
t0 = time.perf_counter()
for i in range(1, n_steps):
    x_full[:, i] = rk4_step(x_full[:, i - 1], A_full, dt)
t_full = time.perf_counter() - t0

# --- ROM: exact modal-exponential update (single matrix-vector product per step) ---
z0 = Phi_inv @ x0.astype(complex)
x_rom = np.zeros((4, n_steps))
x_rom[:, 0] = x0
z = z0.copy()
t0 = time.perf_counter()
for i in range(1, n_steps):
    z = Lambda_d * z
    x_rom[:, i] = np.real(Phi @ z)
t_rom = time.perf_counter() - t0

rmse = np.sqrt(np.mean((x_full - x_rom) ** 2, axis=1))
per_step_full_us = t_full / n_steps * 1e6
per_step_rom_us = t_rom / n_steps * 1e6

print("=== Module A8: Real-Time ROM Aeroelastic Solver ===")
print(f"Simulation: {n_steps} steps at dt={dt*1000:.1f} ms (real-time budget: {dt*1e6:.0f} us/step)")
print(f"Full dense RK4  : {per_step_full_us:.2f} us/step average")
print(f"ROM (modal exp) : {per_step_rom_us:.2f} us/step average "
      f"({per_step_full_us/max(per_step_rom_us,1e-9):.1f}x faster)")
print(f"State RMSE (full vs. ROM), [h, alpha, hdot, alphadot]: {np.round(rmse, 8)}")
budget_us = dt * 1e6
print(f"Real-time budget = {budget_us:.0f} us/step -> "
      f"full model {'MEETS' if per_step_full_us < budget_us else 'MISSES'} budget, "
      f"ROM {'MEETS' if per_step_rom_us < budget_us else 'MISSES'} budget")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
t_arr = np.arange(n_steps) * dt
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].plot(t_arr, x_full[1], color="black", label="full model (RK4)")
axes[0, 0].plot(t_arr, x_rom[1], color="firebrick", ls="--", label="ROM (modal exponential)")
axes[0, 0].set_title("Pitch angle response: full vs. ROM")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("pitch angle [rad]")
axes[0, 0].legend(fontsize=8)

axes[0, 1].semilogy(t_arr, np.abs(x_full[1] - x_rom[1]) + 1e-16, color="darkorange")
axes[0, 1].set_title("Absolute tracking error (pitch)")
axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("|error| [rad]")

axes[1, 0].bar(["Full RK4", "ROM"], [per_step_full_us, per_step_rom_us], color=["firebrick", "seagreen"])
axes[1, 0].axhline(budget_us, color="k", ls="--", label=f"real-time budget ({budget_us:.0f} us)")
axes[1, 0].set_title("Per-step wall-clock cost")
axes[1, 0].set_ylabel("microseconds/step"); axes[1, 0].legend(fontsize=8)

labels = ["h", "alpha", "hdot", "alphadot"]
axes[1, 1].bar(labels, rmse, color="steelblue")
axes[1, 1].set_title("State-wise RMSE (full vs. ROM)")
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].set_yscale("log")

plt.tight_layout()
plt.savefig("outputs/moduleA8_realtime_rom.png", dpi=150)
print("Saved outputs/moduleA8_realtime_rom.png")
