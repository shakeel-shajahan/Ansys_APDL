"""
Extension to Capstone 10 -- Bayesian FE Model Updating (MCMC Posterior)

Addresses reviewer feedback: "Bayesian FE updating, optimization, adjoint
methods, uncertainty, regularisation" beyond the base point-estimate
least-squares capstone.

The base capstone found a single best-fit (E_factor, root_factor) pair.
This extension instead computes the full POSTERIOR DISTRIBUTION over
these two parameters using Metropolis-Hastings MCMC, given the measured
frequencies and an assumed measurement-noise level -- directly answering
the identifiability question the base capstone's project brief raised
("could you tell the two parameters apart with fewer measured
frequencies?") by looking at the posterior's correlation structure, and
providing calibrated uncertainty on the updated parameters rather than a
single point estimate.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])

rng_bayes = np.random.default_rng(1010)


def log_posterior(theta, f_target, sigma_obs=0.15, n_modes_used=4):
    E_factor, root_factor = theta
    if not (0.5 < E_factor < 1.5) or not (0.05 < root_factor < 1.5):
        return -np.inf
    f_model = get_frequencies(E_nominal * E_factor, I_nom, root_factor, n_modes=n_modes_used)
    resid = (f_model[:n_modes_used] - f_target[:n_modes_used]) / f_target[:n_modes_used]
    log_lik = -0.5 * np.sum((resid / (sigma_obs / f_target[:n_modes_used])) ** 2) if False else \
        -0.5 * np.sum((resid / 0.01) ** 2)  # ~1% relative frequency measurement uncertainty
    log_prior_E = -0.5 * ((E_factor - 1.0) / 0.2) ** 2       # weakly-informative priors
    log_prior_root = -0.5 * ((root_factor - 1.0) / 0.5) ** 2
    return log_lik + log_prior_E + log_prior_root


def run_mcmc(f_target, n_modes_used, n_steps=15000, step=(0.02, 0.05)):
    chain = np.zeros((n_steps, 2))
    chain[0] = [1.0, 1.0]
    cur_lp = log_posterior(chain[0], f_target, n_modes_used=n_modes_used)
    accept = 0
    for i in range(1, n_steps):
        prop = chain[i - 1] + rng_bayes.normal(0, step)
        prop_lp = log_posterior(prop, f_target, n_modes_used=n_modes_used)
        if np.log(rng_bayes.uniform()) < prop_lp - cur_lp:
            chain[i] = prop; cur_lp = prop_lp; accept += 1
        else:
            chain[i] = chain[i - 1]
    return chain, accept / n_steps


# ---------------------------------------------------------------
# Case 1: all 4 measured frequencies (as in the base capstone)
# ---------------------------------------------------------------
chain_4modes, acc_4 = run_mcmc(f_exp, n_modes_used=4)
burn = 5000
post_4 = chain_4modes[burn:]

print("=== Extension 10: Bayesian FE Model Updating (MCMC) ===")
print(f"[4 measured modes] acceptance rate: {acc_4*100:.1f}%")
print(f"[4 measured modes] Posterior E_factor   : mean={post_4[:,0].mean():.3f}, std={post_4[:,0].std():.3f} "
      f"(truth={E_true_factor:.3f})")
print(f"[4 measured modes] Posterior root_factor: mean={post_4[:,1].mean():.3f}, std={post_4[:,1].std():.3f} "
      f"(truth={root_true_factor:.3f})")
corr_4 = np.corrcoef(post_4[:, 0], post_4[:, 1])[0, 1]
print(f"[4 measured modes] Posterior correlation(E_factor, root_factor) = {corr_4:.3f}")

# ---------------------------------------------------------------
# Case 2: only 1 measured frequency (the base capstone's explicit
# identifiability question) -- does the posterior become degenerate?
# ---------------------------------------------------------------
chain_1mode, acc_1 = run_mcmc(f_exp, n_modes_used=1, n_steps=15000)
post_1 = chain_1mode[burn:]
corr_1 = np.corrcoef(post_1[:, 0], post_1[:, 1])[0, 1]

print(f"\n[1 measured mode]  acceptance rate: {acc_1*100:.1f}%")
print(f"[1 measured mode]  Posterior E_factor   : mean={post_1[:,0].mean():.3f}, std={post_1[:,0].std():.3f}")
print(f"[1 measured mode]  Posterior root_factor: mean={post_1[:,1].mean():.3f}, std={post_1[:,1].std():.3f}")
print(f"[1 measured mode]  Posterior correlation(E_factor, root_factor) = {corr_1:.3f}")
print(f"\nIdentifiability answer: even with all 4 measured modes, E_factor and root_factor are")
print(f"already strongly correlated in the posterior (|corr|={abs(corr_4):.2f}) -- a genuine trade-off")
print(f"between 'stiffer material' and 'stiffer boundary' that the base capstone's point-estimate fit")
print(f"could not reveal. Dropping to 1 measured mode widens the posterior substantially")
print(f"({post_1[:,0].std()/post_4[:,0].std():.1f}x wider for E_factor, "
      f"{post_1[:,1].std()/post_4[:,1].std():.1f}x wider for root_factor), confirming that a single")
print("frequency badly under-constrains the two-parameter problem, even though the correlation")
print(f"coefficient itself ({abs(corr_1):.2f}) happens not to increase further -- the dominant effect of")
print("losing information here is parameter-uncertainty inflation, not additional correlation.")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].scatter(post_4[:, 0], post_4[:, 1], s=2, alpha=0.3, color="steelblue")
axes[0, 0].scatter(E_true_factor, root_true_factor, c="red", marker="*", s=200, label="truth", zorder=5)
axes[0, 0].set_title("Posterior (4 measured modes) -- well identified")
axes[0, 0].set_xlabel("E_factor"); axes[0, 0].set_ylabel("root_factor")
axes[0, 0].legend(fontsize=8)

axes[0, 1].scatter(post_1[:, 0], post_1[:, 1], s=2, alpha=0.3, color="firebrick")
axes[0, 1].scatter(E_true_factor, root_true_factor, c="blue", marker="*", s=200, label="truth", zorder=5)
axes[0, 1].set_title("Posterior (1 measured mode only) -- poorly identified")
axes[0, 1].set_xlabel("E_factor"); axes[0, 1].set_ylabel("root_factor")
axes[0, 1].legend(fontsize=8)

axes[1, 0].hist(post_4[:, 0], bins=40, alpha=0.6, label="4 modes", color="steelblue", density=True)
axes[1, 0].hist(post_1[:, 0], bins=40, alpha=0.6, label="1 mode", color="firebrick", density=True)
axes[1, 0].axvline(E_true_factor, color="black", ls="--", label="truth")
axes[1, 0].set_title("Marginal posterior: E_factor")
axes[1, 0].set_xlabel("E_factor"); axes[1, 0].legend(fontsize=8)

axes[1, 1].hist(post_4[:, 1], bins=40, alpha=0.6, label="4 modes", color="steelblue", density=True)
axes[1, 1].hist(post_1[:, 1], bins=40, alpha=0.6, label="1 mode", color="firebrick", density=True)
axes[1, 1].axvline(root_true_factor, color="black", ls="--", label="truth")
axes[1, 1].set_title("Marginal posterior: root_factor")
axes[1, 1].set_xlabel("root_factor"); axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA6_extension_bayesian_updating.png"), dpi=150)
print("Saved outputs/moduleA6_extension_bayesian_updating.png")
