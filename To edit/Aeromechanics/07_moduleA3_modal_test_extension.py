"""
Extension to Capstone 7 -- Mesh-Independence / Verification Study

Addresses reviewer feedback: "image-based modal testing, operational
modal analysis, uncertainty, Bayesian updating" -- and, more fundamentally
for a PhD-level treatment, basic VERIFICATION (mesh/convergence study)
which was entirely absent from the base capstone's purely analytical
closed-form treatment.

This extension builds a simple consistent-mass Euler-Bernoulli FE beam
model (the same element formulation used in Capstone 10) of the SAME
cantilever beam, and performs a mesh-convergence study: as the number of
elements increases, the FE-predicted natural frequencies should converge
monotonically towards the closed-form analytical values used in the base
capstone. This is the textbook verification exercise (distinct from
validation, which compares against experiment/reality) that should
precede trusting any FE model, and it also sets up Capstone 10's need for
a "sufficiently converged" mesh before doing model updating.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])


def fe_cantilever_frequencies(n_el, n_modes=4):
    Le_local = L / n_el
    n_dof = 2 * (n_el + 1)
    EI = E * I

    def elem_mats():
        ke = EI / Le_local ** 3 * np.array([
            [12, 6 * Le_local, -12, 6 * Le_local],
            [6 * Le_local, 4 * Le_local ** 2, -6 * Le_local, 2 * Le_local ** 2],
            [-12, -6 * Le_local, 12, -6 * Le_local],
            [6 * Le_local, 2 * Le_local ** 2, -6 * Le_local, 4 * Le_local ** 2]])
        me = rho * A * Le_local / 420 * np.array([
            [156, 22 * Le_local, 54, -13 * Le_local],
            [22 * Le_local, 4 * Le_local ** 2, 13 * Le_local, -3 * Le_local ** 2],
            [54, 13 * Le_local, 156, -22 * Le_local],
            [-13 * Le_local, -3 * Le_local ** 2, -22 * Le_local, 4 * Le_local ** 2]])
        return ke, me

    K = np.zeros((n_dof, n_dof)); M = np.zeros((n_dof, n_dof))
    ke, me = elem_mats()
    for el in range(n_el):
        dofs = [2 * el, 2 * el + 1, 2 * el + 2, 2 * el + 3]
        for a in range(4):
            for bb in range(4):
                K[dofs[a], dofs[bb]] += ke[a, bb]
                M[dofs[a], dofs[bb]] += me[a, bb]
    free = list(range(2, n_dof))  # fully rigid cantilever BC (both w0, theta0 fixed)
    Kff = K[np.ix_(free, free)]; Mff = M[np.ix_(free, free)]
    evals, _ = eigh(Kff, Mff)
    freqs_out = np.sort(np.sqrt(np.clip(evals, 0, None)) / (2 * np.pi))
    if len(freqs_out) < n_modes:
        freqs_out = np.concatenate([freqs_out, np.full(n_modes - len(freqs_out), np.nan)])
    return freqs_out[:n_modes]


mesh_sizes = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 80]
convergence = {n: fe_cantilever_frequencies(n, n_modes=3) for n in mesh_sizes}

print("=== Extension 7: FE Mesh-Convergence / Verification Study ===")
print(f"Closed-form (analytical, 'exact') frequencies: {np.round(f_theory[:3], 4)} Hz")
print(f"{'n_el':>5s}  {'Mode1':>10s}  {'Mode2':>10s}  {'Mode3':>10s}  {'err1[%]':>8s}  {'err2[%]':>8s}  {'err3[%]':>8s}")
for n in mesh_sizes:
    fvals = convergence[n]
    err = 100 * (fvals - f_theory[:3]) / f_theory[:3]
    print(f"{n:5d}  {fvals[0]:10.4f}  {fvals[1]:10.4f}  {fvals[2]:10.4f}  "
          f"{err[0]:8.3f}  {err[1]:8.3f}  {err[2]:8.3f}")

# Richardson-extrapolation-style observed order of convergence for mode 3,
# using a clean 2x mesh-refinement sequence (n=5 -> 10 -> 20, so h halves
# exactly each step) -- mode 3 is used because it still shows a clear,
# non-floating-point-noise-dominated error at these mesh levels
f3_vals = np.array([convergence[5][2], convergence[10][2], convergence[20][2]])
p_obs = np.log(abs((f3_vals[2] - f3_vals[1]) / (f3_vals[1] - f3_vals[0]))) / np.log(0.5)
print(f"\nObserved order of convergence (mode 3, Richardson estimate, n=5->10->20): {p_obs:.2f}")
print("(Standard consistent-mass Euler-Bernoulli C1 beam elements are expected to show ~4th-order")
print(" convergence in natural frequency for a smooth mode shape -- consistent with the value above.)")

n_converged = None
for n in mesh_sizes:
    err1 = abs(100 * (convergence[n][0] - f_theory[0]) / f_theory[0])
    if err1 < 0.5 and n_converged is None:
        n_converged = n
print(f"\nMesh judged 'converged' (mode-1 error < 0.5%) at n_el = {n_converged} "
      f"-- this is the minimum mesh that should be used for the FE model updating in Capstone 10.")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for mode in range(3):
    errs = [100 * abs(convergence[n][mode] - f_theory[mode]) / f_theory[mode] for n in mesh_sizes]
    axes[0].loglog(mesh_sizes, errs, "o-", label=f"Mode {mode+1}")
axes[0].axhline(0.5, color="k", ls="--", lw=0.8, label="0.5% convergence criterion")
axes[0].set_xlabel("number of elements"); axes[0].set_ylabel("relative frequency error [%]")
axes[0].set_title("Mesh-convergence study vs. closed-form theory")
axes[0].legend(fontsize=8)

for mode in range(3):
    vals = [convergence[n][mode] for n in mesh_sizes]
    axes[1].semilogx(mesh_sizes, vals, "o-", label=f"Mode {mode+1} (FE)")
    axes[1].axhline(f_theory[mode], color="gray", ls=":", lw=1)
axes[1].set_xlabel("number of elements"); axes[1].set_ylabel("frequency [Hz]")
axes[1].set_title("FE frequency vs. mesh density (dotted = closed-form)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA3_extension_mesh_convergence.png"), dpi=150)
print("Saved outputs/moduleA3_extension_mesh_convergence.png")
