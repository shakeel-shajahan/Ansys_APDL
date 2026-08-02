"""
Extension to Capstone 4 -- Bayesian (GP) Map Reconstruction with Sensor-
Placement Optimisation

Addresses reviewer feedback: "uncertainty bands, Bayesian interpolation,
sensor placement optimisation" beyond the base RBF + bootstrap capstone.

Part A: replace the deterministic RBF reconstruction with a Gaussian-
Process regression, which gives a principled, closed-form predictive
uncertainty at every point (rather than needing a bootstrap over RBF
refits), and cross-check that the two uncertainty estimates broadly
agree.

Part B: sensor-placement optimisation -- given a budget of 5 additional
instrumented points (one more per speed line), choose their locations to
maximise the reduction in surge-margin uncertainty at the operability-
critical speed identified in the base capstone, using a greedy
max-variance placement strategy, and compare against 5 randomly placed
points.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])


class MapGP:
    def __init__(self, noise=1e-2):
        self.noise = noise

    def _kernel(self, A, B, ell, sigma_f):
        d2 = cdist(A / ell, B / ell, metric="sqeuclidean")
        return sigma_f ** 2 * np.exp(-0.5 * d2)

    def fit(self, X, y):
        self.x_mean = X.mean(axis=0); self.x_std = X.std(axis=0) + 1e-9
        self.X = (X - self.x_mean) / self.x_std
        self.y_mean = y.mean(); self.y = y - self.y_mean
        best = None
        for ell0 in [0.3, 0.7, 1.5]:
            def nll(theta):
                ell = np.exp(theta[:2]); sigma_f = np.exp(theta[2])
                K = self._kernel(self.X, self.X, ell, sigma_f) + self.noise ** 2 * np.eye(len(self.X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e6
                a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                return 0.5 * self.y @ a + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y) * np.log(2 * np.pi)
            x0 = np.log([ell0, ell0, np.std(self.y) + 1e-3])
            res = minimize(nll, x0, method="L-BFGS-B",
                            bounds=[(np.log(0.1), np.log(8)), (np.log(0.1), np.log(8)), (np.log(1e-3), np.log(20))])
            if best is None or res.fun < best.fun:
                best = res
        self.ell = np.exp(best.x[:2]); self.sigma_f = np.exp(best.x[2])
        K = self._kernel(self.X, self.X, self.ell, self.sigma_f) + self.noise ** 2 * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        Xn = (Xs - self.x_mean) / self.x_std
        Ks = self._kernel(Xn, self.X, self.ell, self.sigma_f)
        mean = Ks @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, Ks.T)
        var = np.clip(self.sigma_f ** 2 - np.sum(v ** 2, axis=0), 1e-10, None)
        return mean, np.sqrt(var)


gp_map = MapGP(noise=1e-2).fit(X_meas, pr_meas)
Xg_flat = np.column_stack([MM.ravel(), SS.ravel()])
gp_mean, gp_std = gp_map.predict(Xg_flat)
gp_mean = gp_mean.reshape(MM.shape)
gp_std_2d = gp_std.reshape(MM.shape)

rmse_gp_vs_true = np.sqrt(np.mean((gp_mean - true_map(MM, SS)) ** 2))
rmse_rbf_vs_true = np.sqrt(np.mean((PR - true_map(MM, SS)) ** 2))
print("=== Extension 4: Bayesian (GP) Reconstruction + Sensor Placement ===")
print(f"RBF reconstruction RMSE vs. true map: {rmse_rbf_vs_true:.4f}")
print(f"GP  reconstruction RMSE vs. true map: {rmse_gp_vs_true:.4f}")
print(f"GP length-scales (mdot, speed): {np.round(gp_map.ell, 3)}")

# ---------------------------------------------------------------
# Part B: greedy max-variance sensor placement vs. random placement
# ---------------------------------------------------------------
candidate_speeds = np.array([0.7, 0.8, 0.9, 1.0, 1.05])  # one more point per existing speed line
rng4 = np.random.default_rng(44)


def greedy_placement(gp_model, X_existing, y_existing, n_new=5):
    X_cur = X_existing.copy(); y_cur = y_existing.copy()
    chosen = []
    for _ in range(n_new):
        best_var, best_pt = -1, None
        for s in candidate_speeds:
            center = 0.45 + 0.5 * s
            for m_cand in np.linspace(center - 0.16, center + 0.10, 15):
                _, std_here = gp_model.predict(np.array([[m_cand, s]]))
                if std_here[0] > best_var:
                    best_var, best_pt = std_here[0], (m_cand, s)
        chosen.append(best_pt)
        y_new = true_map(best_pt[0], best_pt[1]) + rng4.normal(0, 0.01)
        X_cur = np.vstack([X_cur, best_pt]); y_cur = np.append(y_cur, y_new)
        gp_model = MapGP(noise=1e-2).fit(X_cur, y_cur)
    return chosen, gp_model


chosen_greedy, gp_after_greedy = greedy_placement(gp_map, X_meas, pr_meas, n_new=5)

# random baseline: same number of new points, random locations on the same speed lines
X_rand = X_meas.copy(); y_rand = pr_meas.copy()
for s in candidate_speeds:
    center = 0.45 + 0.5 * s
    m_cand = rng4.uniform(center - 0.16, center + 0.10)
    y_new = true_map(m_cand, s) + rng4.normal(0, 0.01)
    X_rand = np.vstack([X_rand, [m_cand, s]]); y_rand = np.append(y_rand, y_new)
gp_after_random = MapGP(noise=1e-2).fit(X_rand, y_rand)

mean_greedy, std_greedy = gp_after_greedy.predict(Xg_flat)
mean_random, std_random = gp_after_random.predict(Xg_flat)

print(f"\nMean map-wide predictive std BEFORE 5 new points : {gp_std.mean():.4f}")
print(f"Mean map-wide predictive std AFTER greedy placement: {std_greedy.mean():.4f} "
      f"({100*(1-std_greedy.mean()/gp_std.mean()):.1f}% reduction)")
print(f"Mean map-wide predictive std AFTER random placement: {std_random.mean():.4f} "
      f"({100*(1-std_random.mean()/gp_std.mean()):.1f}% reduction)")
print(f"\nGreedy-chosen sensor locations (mdot, speed): "
      f"{[(round(a,3), round(b,3)) for a,b in chosen_greedy]}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

c0 = axes[0, 0].contourf(MM, SS, gp_mean, levels=20, cmap="viridis")
axes[0, 0].scatter(mdot_meas, speed_meas, c="white", edgecolor="k", s=20)
axes[0, 0].set_title("GP reconstruction (with closed-form uncertainty)")
axes[0, 0].set_xlabel("corrected mass flow"); axes[0, 0].set_ylabel("corrected speed")
plt.colorbar(c0, ax=axes[0, 0])

c1 = axes[0, 1].contourf(MM, SS, gp_std_2d, levels=20, cmap="magma")
axes[0, 1].scatter(mdot_meas, speed_meas, c="cyan", edgecolor="k", s=20, label="existing 30 points")
gx = [p[0] for p in chosen_greedy]; gy = [p[1] for p in chosen_greedy]
axes[0, 1].scatter(gx, gy, c="lime", edgecolor="k", s=80, marker="*", label="greedy new points")
axes[0, 1].set_title("GP predictive std. + greedy sensor placement")
axes[0, 1].set_xlabel("corrected mass flow"); axes[0, 1].set_ylabel("corrected speed")
axes[0, 1].legend(fontsize=7)
plt.colorbar(c1, ax=axes[0, 1])

axes[1, 0].bar(["before", "after greedy\nplacement", "after random\nplacement"],
               [gp_std.mean(), std_greedy.mean(), std_random.mean()],
               color=["gray", "seagreen", "firebrick"])
axes[1, 0].set_title("Map-wide mean predictive std. (5 new points)")
axes[1, 0].set_ylabel("mean predictive std")

axes[1, 1].bar(["RBF\n(base capstone)", "GP\n(this extension)"],
               [rmse_rbf_vs_true, rmse_gp_vs_true], color=["firebrick", "seagreen"])
axes[1, 1].set_title("Reconstruction RMSE vs. true map")
axes[1, 1].set_ylabel("pressure-ratio RMSE")

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleD4_extension_gp_sensor_placement.png"), dpi=150)
print("Saved outputs/moduleD4_extension_gp_sensor_placement.png")
