"""
Capstone 10 -- Finite-Element Model Updating from Modal Test Data

Goal
----
A common gap in practice: the as-built FE model of a structure never
matches the physical test article exactly (uncertain boundary stiffness,
material properties, manufacturing tolerances). Model updating adjusts a
small number of physically meaningful FE parameters so the model's
predicted modal frequencies match the experimentally identified ones
(e.g. from Module A1/A3) -- exactly the workflow needed before trusting
an FE model for downstream stress/aeromechanical predictions.

Method
------
1. A 10-element Euler-Bernoulli cantilever beam FE model (consistent mass
   + stiffness matrices), with an *unknown* root stiffness reduction
   factor (loose clamping) and an *unknown* Young's modulus scaling.
2. "Truth": a specific pair of parameters generates the "experimental"
   frequencies (with small measurement noise) -- this stands in for the
   4 modal frequencies you would get from an actual test.
3. "As-built" (uncalibrated) baseline model uses nominal (wrong)
   parameters and clearly disagrees with test.
4. Model updating: nonlinear least-squares over the 2 parameters to
   minimize the sum-of-squares error between FE-predicted and
   "measured" frequencies (gradient-free, robust Nelder-Mead).
5. Report before/after frequency errors and the recovered parameters vs.
   the (in a real test, unknown-but-here-known) truth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh

# ---------------------------------------------------------------
# 1. FE model: n_el Euler-Bernoulli beam elements, cantilevered
# ---------------------------------------------------------------
n_el = 10
L_total = 0.30
Le = L_total / n_el
E_nominal = 70e9
rho = 2700.0
width = 0.02
thick = 0.003
A = width * thick
I_nom = width * thick ** 3 / 12
n_dof = 2 * (n_el + 1)  # (w, theta) per node


def element_matrices(E, I):
    EI = E * I
    ke = EI / Le ** 3 * np.array([
        [12, 6 * Le, -12, 6 * Le],
        [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
        [-12, -6 * Le, 12, -6 * Le],
        [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]])
    me = rho * A * Le / 420 * np.array([
        [156, 22 * Le, 54, -13 * Le],
        [22 * Le, 4 * Le ** 2, 13 * Le, -3 * Le ** 2],
        [54, 13 * Le, 156, -22 * Le],
        [-13 * Le, -3 * Le ** 2, -22 * Le, 4 * Le ** 2]])
    return ke, me


def assemble(E, I, root_stiff_factor):
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    for el in range(n_el):
        ke, me = element_matrices(E, I)
        dofs = [2 * el, 2 * el + 1, 2 * el + 2, 2 * el + 3]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]
                M[dofs[a], dofs[b]] += me[a, b]
    # Cantilever BC: the root TRANSLATION (w0, dof 0) is rigidly clamped (removed).
    # The root ROTATION (theta0, dof 1) is retained and restrained by a finite
    # rotational spring -- this represents an uncertain bolted/flanged root
    # stiffness rather than an idealised perfectly-rigid clamp.
    k_root_nominal = 300.0  # N*m/rad; calibrated so root flexibility measurably
    #                         shifts frequencies without swamping the beam's own stiffness
    free = list(range(1, n_dof))  # keep theta0 (now local index 0) and everything else
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]
    Kff[0, 0] += k_root_nominal * root_stiff_factor
    return Kff, Mff


def get_frequencies(E, I, root_stiff_factor, n_modes=4):
    Kff, Mff = assemble(E, I, root_stiff_factor)
    evals, _ = eigh(Kff, Mff)
    evals = np.clip(evals, 0, None)
    wn = np.sqrt(evals)
    f_hz = wn / (2 * np.pi)
    return np.sort(f_hz)[:n_modes]


# ---------------------------------------------------------------
# 2. "Truth" parameters (unknown to the updating process) generate the
#    pseudo-experimental frequencies
# ---------------------------------------------------------------
E_true_factor = 0.93     # actual E is 93% of the nominal datasheet value
root_true_factor = 0.55  # the root clamp is significantly looser than assumed

rng = np.random.default_rng(6)
f_exp_true = get_frequencies(E_nominal * E_true_factor, I_nom, root_true_factor)
f_exp = f_exp_true * (1 + rng.normal(0, 0.003, len(f_exp_true)))  # small test/identification noise

# ---------------------------------------------------------------
# 3. Baseline "as-built" (uncalibrated) FE model -- nominal, wrong parameters
# ---------------------------------------------------------------
f_baseline = get_frequencies(E_nominal, I_nom, 1.0)

# ---------------------------------------------------------------
# 4. Model updating: least-squares over (E_factor, root_factor)
# ---------------------------------------------------------------
def objective(theta):
    E_factor, root_factor = theta
    if E_factor <= 0.3 or E_factor > 1.5 or root_factor <= 0.05 or root_factor > 1.5:
        return 1e6
    f_model = get_frequencies(E_nominal * E_factor, I_nom, root_factor)
    return np.sum(((f_model - f_exp) / f_exp) ** 2)


res = minimize(objective, x0=[1.0, 1.0], method="Nelder-Mead",
               options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 500})
E_factor_updated, root_factor_updated = res.x
f_updated = get_frequencies(E_nominal * E_factor_updated, I_nom, root_factor_updated)

print("=== Module A6: FE Model Updating from Modal Test Data ===")
print(f"'Experimental' frequencies [Hz]      : {np.round(f_exp, 2)}")
print(f"Baseline (as-built) FE frequencies    : {np.round(f_baseline, 2)}   "
      f"(max error {100*np.max(np.abs(f_baseline-f_exp)/f_exp):.1f}%)")
print(f"Updated FE frequencies                : {np.round(f_updated, 2)}   "
      f"(max error {100*np.max(np.abs(f_updated-f_exp)/f_exp):.1f}%)")
print(f"Recovered E factor   : {E_factor_updated:.3f}  (truth = {E_true_factor:.3f})")
print(f"Recovered root factor: {root_factor_updated:.3f}  (truth = {root_true_factor:.3f})")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

x = np.arange(len(f_exp))
w = 0.25
axes[0].bar(x - w, f_exp, width=w, label="'measured' (target)", color="black")
axes[0].bar(x, f_baseline, width=w, label="baseline FE (as-built)", color="firebrick")
axes[0].bar(x + w, f_updated, width=w, label="updated FE", color="seagreen")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"Mode {i+1}" for i in x])
axes[0].set_ylabel("frequency [Hz]")
axes[0].set_title("Baseline vs. updated FE frequencies")
axes[0].legend(fontsize=8)

err_baseline = 100 * (f_baseline - f_exp) / f_exp
err_updated = 100 * (f_updated - f_exp) / f_exp
axes[1].bar(x - 0.15, err_baseline, width=0.3, label="baseline error", color="firebrick")
axes[1].bar(x + 0.15, err_updated, width=0.3, label="updated error", color="seagreen")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([f"Mode {i+1}" for i in x])
axes[1].set_ylabel("frequency error [%]")
axes[1].set_title("Model-updating error reduction")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/moduleA6_fe_updating.png", dpi=150)
print("Saved outputs/moduleA6_fe_updating.png")
