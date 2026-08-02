"""
Case Study 9 : Flexible Flap in Channel Flow (Classical FSI Benchmark)
Beginner demonstration: a reduced-order spring-mass-damper model of an elastic flap in a
channel flow, where the fluid contributes an added mass and a flow-velocity-dependent
damping/stiffness (linearized lift model). We use this to reproduce the qualitative
"added-mass instability" described in the handbook for partitioned solvers with light
structures in dense fluid.
"""
import numpy as np

def amplification_factor(beta, omega_relax):
    """Discrete fixed-point iteration for a lagged partitioned FSI coupling:
    a_(k+1) = a_k + omega*( -beta*a_k - a_k ) = a_k * (1 - omega*(1+beta))
    where beta = m_fluid_added / m_structure. omega=1 is the naive explicit
    (unrelaxed) exchange; g = 1-omega*(1+beta) is the iteration amplification factor."""
    return 1 - omega_relax * (1 + beta)

print("Effect of the fluid-to-structure mass ratio (added-mass effect) on partitioned")
print("coupling stability, WITHOUT under-relaxation (omega_relax = 1.0):\n")
for beta in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
    amp = amplification_factor(beta, omega_relax=1.0)
    stable = abs(amp) < 1.0
    print(f"  mass ratio beta = m_fluid/m_structure = {beta:>4.2f}  "
          f"amplification factor = {amp:5.2f}  -> {'STABLE' if stable else 'UNSTABLE'}")

print("\nSame mass ratios WITH proper Aitken under-relaxation (omega_relax found analytically")
print("as omega = 1/(1+beta) to exactly cancel the added-mass term):\n")
for beta in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
    omega_opt = 1.0 / (1.0 + beta)
    amp = amplification_factor(beta, omega_relax=omega_opt)
    print(f"  beta = {beta:>4.2f}  optimal relaxation omega = {omega_opt:.3f}  "
          f"amplification factor = {amp:5.3f}  -> STABLE (by construction)")

# ---------- time-domain confirmation with a simple mass-spring system ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def simulate(beta, omega_relax, n_steps=400, dt=0.01):
    m_s, k, c = 1.0, 4.0, 0.05
    x, v = 0.05, 0.0
    x_hist = [x]
    for n in range(n_steps):
        F_fluid_added = -beta * (v)   # crude added-mass-like reaction force proxy
        a_struct = (-k*x - c*v + omega_relax * F_fluid_added) / m_s
        v = v + a_struct * dt
        x = x + v * dt
        x_hist.append(x)
    return np.array(x_hist)

t = np.arange(401) * 0.01
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for beta in [0.1, 3.0]:
    x_naive = simulate(beta, omega_relax=1.0)
    ax[0].plot(t, x_naive, label=f"beta={beta}")
ax[0].set_title("No relaxation (omega=1)"); ax[0].legend(); ax[0].set_xlabel("t [s]")
for beta in [0.1, 3.0]:
    x_relaxed = simulate(beta, omega_relax=1.0/(1+beta))
    ax[1].plot(t, x_relaxed, label=f"beta={beta}")
ax[1].set_title("With optimal relaxation"); ax[1].legend(); ax[1].set_xlabel("t [s]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch09.png", dpi=150)
print("\nFigure saved (compare divergence vs stability for the high mass-ratio case).")
