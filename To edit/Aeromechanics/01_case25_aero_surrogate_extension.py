"""
Extension to Capstone 1 -- Active Learning (Bayesian-Optimization-style)
Next-Point Selection + Sobol Sensitivity Analysis

Addresses reviewer feedback: "active learning, Bayesian optimisation,
uncertainty propagation" were missing from the base GP surrogate capstone.

Part A: Sobol-style variance-based sensitivity analysis (Saltelli sampling,
first-order + total-order indices) on the GP surrogate mean prediction,
to answer: which input (mass flow or speed) drives more of the pressure-
ratio variance across the operating envelope?

Part B: Active learning -- given a budget of ONE more CFD run, choose the
next (mdot, speed) point that maximises the GP's predictive variance
(uncertainty sampling), a standard active-learning acquisition rule, and
compare it against a purely random next-point choice.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Re-use the exact GP class and trained surrogate from the base capstone
exec(open(os.path.join(os.path.dirname(__file__), "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# 5. Plots"
)[0])

rng2 = np.random.default_rng(101)

# ---------------------------------------------------------------
# Part A: Sobol variance-based sensitivity (Saltelli sampling, first &
# total order indices) on the GP surrogate mean for pressure ratio
# ---------------------------------------------------------------
def sobol_indices(model, bounds, n=4096, seed=0):
    """First-order and total-order Sobol indices via Saltelli's sampling scheme
    (Saltelli 2002), estimated directly from the fitted GP surrogate mean."""
    rs = np.random.default_rng(seed)
    d = len(bounds)
    A = rs.uniform(0, 1, (n, d))
    B = rs.uniform(0, 1, (n, d))
    for j in range(d):
        lo, hi = bounds[j]
        A[:, j] = lo + A[:, j] * (hi - lo)
        B[:, j] = lo + B[:, j] * (hi - lo)

    def f(X):
        mean, _ = model.predict(X)
        return mean

    fA = f(A)
    fB = f(B)
    f0 = np.mean(fA)
    var_total = np.var(np.concatenate([fA, fB]))

    S1 = np.zeros(d)
    ST = np.zeros(d)
    for j in range(d):
        AB_j = A.copy()
        AB_j[:, j] = B[:, j]
        f_ABj = f(AB_j)
        # first-order (Saltelli/Jansen estimator)
        S1[j] = np.mean(fB * (f_ABj - fA)) / var_total
        # total-order (Jansen 1999 estimator)
        ST[j] = 0.5 * np.mean((fA - f_ABj) ** 2) / var_total
    return S1, ST


bounds = [(0.35, 1.0), (0.65, 1.05)]  # (mdot, speed) ranges used in the base capstone
S1, ST = sobol_indices(gp_pr, bounds, n=4096, seed=1)
S1_report = np.clip(S1, 0, None)  # small negative S1 values are a known finite-sample
                                    # estimator artifact for near-zero true indices, not a
                                    # negative "true" sensitivity (which is not physical)

print("=== Extension 1a: Sobol Sensitivity Analysis (pressure ratio) ===")
print(f"First-order index  S1 (raw estimator) : mdot={S1[0]:.3f}, speed={S1[1]:.3f}")
print(f"First-order index  S1 (clipped >= 0)  : mdot={S1_report[0]:.3f}, speed={S1_report[1]:.3f}")
print(f"Total-order index  ST: mdot={ST[0]:.3f}, speed={ST[1]:.3f}")
interaction = np.sum(ST) - np.sum(S1)
print(f"Sum(ST) - Sum(S1) = {interaction:.3f}  -> interaction/nonlinearity share of variance")
dominant = "speed" if S1[1] > S1[0] else "mass flow"
print(f"Dominant driver of pressure-ratio variance across the envelope: {dominant}")

# ---------------------------------------------------------------
# Part B: Active learning -- next-point selection by maximum predictive
# variance (uncertainty sampling), vs. a random candidate
# ---------------------------------------------------------------
candidate_speed = rng2.uniform(0.65, 1.05, 2000)
candidate_mdot = np.array([rng2.uniform(0.35 + 0.5 * s, 0.55 + 0.45 * s) for s in candidate_speed])
X_candidates = np.column_stack([candidate_mdot, candidate_speed])

_, pr_std_candidates = gp_pr.predict(X_candidates)
best_idx = np.argmax(pr_std_candidates)
best_point = X_candidates[best_idx]
random_idx = rng2.integers(0, len(X_candidates))
random_point = X_candidates[random_idx]

print("\n=== Extension 1b: Active Learning -- Next CFD Point Selection ===")
print(f"Uncertainty-sampling choice: mdot={best_point[0]:.3f}, speed={best_point[1]:.3f}, "
      f"predicted std={pr_std_candidates[best_idx]:.4f}")
print(f"Random-choice baseline     : mdot={random_point[0]:.3f}, speed={random_point[1]:.3f}, "
      f"predicted std={pr_std_candidates[random_idx]:.4f}")
print(f"Uncertainty-sampling point has "
      f"{pr_std_candidates[best_idx]/max(pr_std_candidates[random_idx],1e-9):.1f}x "
      f"the predictive std of the random candidate -- confirms it targets the least-known region.")

# verify: does adding this point actually reduce map-wide uncertainty more than a random point?
def refit_with_extra_point(extra_point, extra_y):
    X_new = np.vstack([X_train, extra_point])
    y_new = np.append(y_pr, extra_y)
    return SimpleGP(noise=noise_pr).fit(X_new, y_new)


extra_y_best = true_pressure_ratio(best_point[0], best_point[1])
extra_y_rand = true_pressure_ratio(random_point[0], random_point[1])
gp_after_best = refit_with_extra_point(best_point, extra_y_best)
gp_after_rand = refit_with_extra_point(random_point, extra_y_rand)

eval_grid = np.column_stack([rng2.uniform(0.35, 1.0, 500), rng2.uniform(0.65, 1.05, 500)])
_, std_before = gp_pr.predict(eval_grid)
_, std_after_best = gp_after_best.predict(eval_grid)
_, std_after_rand = gp_after_rand.predict(eval_grid)

print(f"\nMean map-wide predictive std BEFORE new point      : {std_before.mean():.5f}")
print(f"Mean map-wide predictive std AFTER uncertainty-pick : {std_after_best.mean():.5f} "
      f"({100*(1-std_after_best.mean()/std_before.mean()):.1f}% reduction)")
print(f"Mean map-wide predictive std AFTER random point     : {std_after_rand.mean():.5f} "
      f"({100*(1-std_after_rand.mean()/std_before.mean()):.2f}% reduction)")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

x = np.arange(2)
w = 0.35
axes[0].bar(x - w/2, S1, width=w, label="first-order $S_1$", color="steelblue")
axes[0].bar(x + w/2, ST, width=w, label="total-order $S_T$", color="firebrick")
axes[0].set_xticks(x); axes[0].set_xticklabels(["mass flow", "speed"])
axes[0].set_title("Sobol sensitivity indices (pressure ratio)")
axes[0].set_ylabel("index value"); axes[0].legend(fontsize=8)

sc = axes[1].scatter(candidate_speed, candidate_mdot, c=pr_std_candidates, cmap="magma", s=4)
axes[1].scatter(best_point[1], best_point[0], c="cyan", edgecolor="k", s=120, marker="*",
                label="uncertainty-sampling pick", zorder=5)
axes[1].scatter(random_point[1], random_point[0], c="white", edgecolor="k", s=80, marker="o",
                label="random baseline", zorder=5)
axes[1].scatter(speed_train, mdot_train, c="lime", edgecolor="k", s=25, label="existing 16 CFD points")
axes[1].set_title("GP predictive std. + next-point candidates")
axes[1].set_xlabel("corrected speed"); axes[1].set_ylabel("corrected mass flow")
axes[1].legend(fontsize=7)
plt.colorbar(sc, ax=axes[1])

axes[2].bar(["before", "after\nuncertainty pick", "after\nrandom pick"],
            [std_before.mean(), std_after_best.mean(), std_after_rand.mean()],
            color=["gray", "seagreen", "firebrick"])
axes[2].set_title("Map-wide mean predictive std.\n(active learning value)")
axes[2].set_ylabel("mean predictive std")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "outputs", "case25_extension_sobol_active_learning.png"), dpi=150)
print("\nSaved outputs/case25_extension_sobol_active_learning.png")
