"""
Capstone 1 -- Calibrated Aerodynamic Surrogate for a Transonic Axial Rotor
(inspired by NASA Rotor 37 style CFD sweeps)

Goal
----
Build a Gaussian-Process surrogate that predicts total-to-total pressure
ratio and adiabatic efficiency of a transonic axial compressor/turbine
rotor stage as a function of (corrected mass flow, corrected speed), using
a sparse set of "CFD" operating points. Quantify predictive uncertainty and
show honestly where the surrogate can and cannot be trusted (interpolation
vs extrapolation), which is exactly the aerodynamic-design question a
last-stage LP steam turbine rotor faces when only a handful of expensive
3D RANS runs are affordable.

Method
------
1. Synthetic "CFD" data generator: a smooth performance-map function with
   realistic curvature (pressure ratio rises with speed, falls off near
   choke; efficiency has an island peak) plus small CFD-run-to-run noise.
2. Gaussian Process Regression (scikit-learn-free, using scipy) with a
   squared-exponential kernel, fit on a coarse Latin-Hypercube-like design.
3. Predict on a dense grid; compare against ground truth in-envelope and
   just outside the training envelope (extrapolation stress test).
4. Report RMSE/coverage in-envelope vs out-of-envelope.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

rng = np.random.default_rng(25)

# ---------------------------------------------------------------
# 1. Ground-truth synthetic performance map (unknown to the surrogate)
# ---------------------------------------------------------------
def true_pressure_ratio(mdot, speed):
    """mdot in [0.6,1.05] corrected mass flow fraction, speed in [0.6,1.05] corrected speed."""
    choke_penalty = np.exp(-((mdot - (0.55 + 0.42 * speed)) / 0.18) ** 2 * 0.0) # placeholder unused
    base = 1.15 + 0.85 * speed ** 1.8 - 0.55 * (mdot - (0.35 + 0.55 * speed)) ** 2 / (0.05 + 0.25 * speed)
    return np.clip(base, 1.0, None)

def true_efficiency(mdot, speed):
    center_m = 0.4 + 0.5 * speed
    eff = 0.90 - 6.0 * (mdot - center_m) ** 2 - 0.05 * (1.0 - speed) ** 2
    return np.clip(eff, 0.3, 0.93)

# ---------------------------------------------------------------
# 2. Sparse "CFD" training design (16 expensive runs -- realistic budget)
# ---------------------------------------------------------------
n_train = 16
speed_train = rng.uniform(0.65, 1.0, n_train)
mdot_train = np.array([rng.uniform(0.35 + 0.5 * s, 0.55 + 0.45 * s) for s in speed_train])
X_train = np.column_stack([mdot_train, speed_train])

noise_pr = 0.01
noise_eff = 0.006
y_pr = true_pressure_ratio(mdot_train, speed_train) + rng.normal(0, noise_pr, n_train)
y_eff = true_efficiency(mdot_train, speed_train) + rng.normal(0, noise_eff, n_train)


# ---------------------------------------------------------------
# 3. Gaussian Process regression (manual implementation)
# ---------------------------------------------------------------
class SimpleGP:
    def __init__(self, noise=1e-3):
        self.noise = noise

    def _kernel(self, A, B, ell, sigma_f):
        d2 = cdist(A / ell, B / ell, metric="sqeuclidean")
        return sigma_f ** 2 * np.exp(-0.5 * d2)

    def fit(self, X, y):
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-9
        self.X = (X - self.x_mean) / self.x_std
        self.y_mean = y.mean()
        self.y = y - self.y_mean

        best = None
        for ell0 in [0.3, 0.6, 1.0, 2.0]:
            def neg_log_marg(theta):
                ell = np.exp(theta[:2])
                sigma_f = np.exp(theta[2])
                K = self._kernel(self.X, self.X, ell, sigma_f) + self.noise ** 2 * np.eye(len(self.X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e6
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                nll = 0.5 * self.y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y) * np.log(2 * np.pi)
                return nll
            x0 = np.log([ell0, ell0, np.std(self.y) + 1e-3])
            res = minimize(neg_log_marg, x0, method="L-BFGS-B",
                            bounds=[(np.log(0.05), np.log(5)), (np.log(0.05), np.log(5)),
                                    (np.log(1e-3), np.log(10))])
            if best is None or res.fun < best.fun:
                best = res
        self.ell = np.exp(best.x[:2])
        self.sigma_f = np.exp(best.x[2])
        K = self._kernel(self.X, self.X, self.ell, self.sigma_f) + self.noise ** 2 * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        Xs_n = (Xs - self.x_mean) / self.x_std
        Ks = self._kernel(Xs_n, self.X, self.ell, self.sigma_f)
        mean = Ks @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, Ks.T)
        Kss = self.sigma_f ** 2 * np.ones(len(Xs_n))
        var = Kss - np.sum(v ** 2, axis=0)
        var = np.clip(var, 1e-10, None)
        return mean, np.sqrt(var)


gp_pr = SimpleGP(noise=noise_pr).fit(X_train, y_pr)
gp_eff = SimpleGP(noise=noise_eff).fit(X_train, y_eff)

# ---------------------------------------------------------------
# 4. Dense evaluation grid: split into "in-envelope" (convex hull of
#    training design) and "extrapolation" (beyond it) regions
# ---------------------------------------------------------------
speed_grid = np.linspace(0.55, 1.05, 60)
mdot_grid = np.linspace(0.3, 1.0, 60)
SS, MM = np.meshgrid(speed_grid, mdot_grid)
Xg = np.column_stack([MM.ravel(), SS.ravel()])

pr_mean, pr_std = gp_pr.predict(Xg)
eff_mean, eff_std = gp_eff.predict(Xg)
pr_true = true_pressure_ratio(Xg[:, 0], Xg[:, 1])
eff_true = true_efficiency(Xg[:, 0], Xg[:, 1])

train_speed_min, train_speed_max = speed_train.min(), speed_train.max()
in_env = (Xg[:, 1] >= train_speed_min) & (Xg[:, 1] <= train_speed_max)

rmse_in = np.sqrt(np.mean((pr_mean[in_env] - pr_true[in_env]) ** 2))
rmse_out = np.sqrt(np.mean((pr_mean[~in_env] - pr_true[~in_env]) ** 2))
z = np.abs(pr_mean - pr_true) / pr_std
coverage_in = np.mean(z[in_env] <= 1.96)
coverage_out = np.mean(z[~in_env] <= 1.96)

print("=== Case 25: Aerodynamic Surrogate ===")
print(f"GP length-scales (mdot, speed): {gp_pr.ell}")
print(f"Pressure-ratio RMSE  in-envelope : {rmse_in:.4f}")
print(f"Pressure-ratio RMSE  extrapolated: {rmse_out:.4f}  (>{rmse_in/max(rmse_out,1e-9):.2f}x worse ratio)")
print(f"95% interval coverage in-envelope : {coverage_in*100:.1f}%")
print(f"95% interval coverage extrapolated: {coverage_out*100:.1f}%")

# ---------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

c0 = axes[0, 0].contourf(SS, MM, pr_true.reshape(SS.shape), levels=20, cmap="viridis")
axes[0, 0].scatter(speed_train, mdot_train, c="red", edgecolor="white", s=40, label="CFD training points")
axes[0, 0].set_title("Ground-truth pressure ratio map")
axes[0, 0].set_xlabel("corrected speed"); axes[0, 0].set_ylabel("corrected mass flow")
axes[0, 0].legend(loc="upper left", fontsize=8)
plt.colorbar(c0, ax=axes[0, 0])

c1 = axes[0, 1].contourf(SS, MM, pr_mean.reshape(SS.shape), levels=20, cmap="viridis")
axes[0, 1].scatter(speed_train, mdot_train, c="red", edgecolor="white", s=40)
axes[0, 1].axvline(train_speed_min, color="k", ls="--", lw=1)
axes[0, 1].axvline(train_speed_max, color="k", ls="--", lw=1, label="training speed envelope")
axes[0, 1].set_title("GP surrogate mean prediction")
axes[0, 1].set_xlabel("corrected speed"); axes[0, 1].set_ylabel("corrected mass flow")
axes[0, 1].legend(loc="upper left", fontsize=8)
plt.colorbar(c1, ax=axes[0, 1])

c2 = axes[1, 0].contourf(SS, MM, pr_std.reshape(SS.shape), levels=20, cmap="magma")
axes[1, 0].scatter(speed_train, mdot_train, c="cyan", edgecolor="k", s=40)
axes[1, 0].set_title("GP predictive std. dev. (epistemic uncertainty)")
axes[1, 0].set_xlabel("corrected speed"); axes[1, 0].set_ylabel("corrected mass flow")
plt.colorbar(c2, ax=axes[1, 0])

axes[1, 1].bar(["RMSE\nin-envelope", "RMSE\nextrapolated"], [rmse_in, rmse_out], color=["seagreen", "firebrick"])
axes[1, 1].set_title("Interpolation vs. extrapolation error")
axes[1, 1].set_ylabel("pressure-ratio RMSE")
for i, v in enumerate([rmse_in, rmse_out]):
    axes[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("outputs/case25_surrogate_summary.png", dpi=150)
print("Saved outputs/case25_surrogate_summary.png")
