"""
Extension to Capstone 8 -- Covariance-Driven Stochastic Subspace
Identification (SSI-COV)

Addresses reviewer feedback: "SSI, Bayesian OMA, damage localisation,
mode tracking" beyond the base FDD-only capstone.

SSI-COV is the other major family of OMA algorithms (alongside FDD),
working in the TIME domain via output covariances rather than the
frequency domain. It builds a block-Hankel/Toeplitz matrix of output
covariances, extracts an observability-matrix estimate via SVD, and
recovers a discrete-time state-space model whose eigenvalues give modal
frequencies and damping ratios (something FDD, as implemented in the
base capstone, does NOT give -- FDD there only gave frequencies and
shapes, not damping). This lets us directly answer the reviewer's
implicit question: does an independent, time-domain method corroborate
the frequency-domain FDD identification, and can it additionally recover
damping ratios and mode-track them?
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


def ssi_cov(y, fs, n_lags=60, order=6):
    """Covariance-driven SSI. y: (n_channels, n_samples) output data.
    Returns identified natural frequencies [Hz] and damping ratios."""
    n_ch = y.shape[0]
    y = y - y.mean(axis=1, keepdims=True)
    # block Toeplitz matrix of output covariances R_i = E[y(k+i) y(k)^T]
    N = y.shape[1]
    R = np.zeros((n_lags, n_ch, n_ch))
    for i in range(n_lags):
        R[i] = (y[:, i:] @ y[:, :N - i].T) / (N - i)

    # Build block Toeplitz (Hankel-of-covariances) matrix T1
    n_block_rows = n_lags // 2
    T1 = np.zeros((n_block_rows * n_ch, n_block_rows * n_ch))
    for i in range(n_block_rows):
        for j in range(n_block_rows):
            lag = i + j + 1
            if lag < n_lags:
                T1[i * n_ch:(i + 1) * n_ch, j * n_ch:(j + 1) * n_ch] = R[lag]

    U, S, Vt = np.linalg.svd(T1)
    order = min(order, len(S))
    U1 = U[:, :order]
    S1 = np.diag(np.sqrt(S[:order]))
    Obs = U1 @ S1  # extended observability matrix estimate

    C_id = Obs[:n_ch, :]
    Obs_up = Obs[:-n_ch, :]
    Obs_down = Obs[n_ch:, :]
    A_id, *_ = np.linalg.lstsq(Obs_up, Obs_down, rcond=None)

    eigvals_d = np.linalg.eigvals(A_id)
    eigvals_d = eigvals_d[np.abs(eigvals_d) < 1.0]  # keep stable discrete poles
    eigvals_d = eigvals_d[eigvals_d.imag > 1e-8]
    lam_c = np.log(eigvals_d) * fs  # continuous-time poles
    wn = np.abs(lam_c)
    freqs_hz = wn / (2 * np.pi)
    zeta = -lam_c.real / wn
    order_idx = np.argsort(freqs_hz)
    return freqs_hz[order_idx], zeta[order_idx]


print("=== Extension 8: Covariance-Driven Stochastic Subspace Identification (SSI-COV) ===")
freqs_ssi, zeta_ssi = ssi_cov(resp_meas, fs, n_lags=80, order=8)
freqs_ssi = freqs_ssi[(freqs_ssi > 0.3) & (freqs_ssi < 20)]
print(f"True modal frequencies [Hz]           : {np.round(wn_true_hz, 3)}")
print(f"SSI-COV identified frequencies [Hz]   : {np.round(freqs_ssi[:5], 3)}")
print(f"SSI-COV identified damping ratios      : {np.round(zeta_ssi[:len(freqs_ssi[:5])], 4)}")
print(f"(True damping ratio was {c_ratio:.3f} stiffness-proportional -- SSI-COV recovers damping,")
print(" which the base capstone's FDD implementation did not provide at all.)")
print("\nHonest caveat: at model order 8 the identified pole set includes at least one spurious")
print("computational mode not corresponding to any true structural mode (a well-known SSI artifact),")
print("and the identified damping ratios show a noticeable high bias relative to the true 0.6%")
print("stiffness-proportional damping. A production SSI-COV workflow would resolve both issues with")
print("a stabilization diagram across increasing model order (see Capstone 5's Extension) to reject")
print("spurious poles and average damping estimates over the stable pole cluster -- this extension")
print("demonstrates the core SSI-COV mechanics, not a fully hardened, production-grade pipeline.")

# Mode-tracking demonstration: run SSI-COV on 5 independent ambient records
# (different random seeds) and check frequency consistency run-to-run
print("\nMode-tracking across 5 independent ambient records (different noise realisations):")
tracked = []
for seed in range(5):
    rng_track = np.random.default_rng(1000 + seed)
    force_track = np.zeros((3, n))
    base_noise_t = rng_track.normal(0, 1.0, n)
    for i in range(3):
        corr = 0.4
        force_track[i] = corr * base_noise_t + np.sqrt(1 - corr ** 2) * rng_track.normal(0, 1.0, n)
    force_track *= np.array([[1.3], [1.0], [0.6]])

    def rhs_t(ti, y_):
        x_ = y_[:3]; v_ = y_[3:]
        idx = min(int(ti / force_dt), n - 1)
        f_ = force_track[:, idx]
        a_ = Minv @ (f_ - C @ v_ - K @ x_)
        return np.concatenate([v_, a_])

    sol_t = solve_ivp(rhs_t, [0, T], np.zeros(6), t_eval=t, max_step=1 / fs)
    resp_t = sol_t.y[:3] + rng_track.normal(0, 0.02 * sol_t.y[:3].std(), (3, n))
    f_run, _ = ssi_cov(resp_t, fs, n_lags=80, order=8)
    f_run = f_run[(f_run > 0.3) & (f_run < 20)]
    tracked.append(f_run[:3] if len(f_run) >= 3 else np.pad(f_run, (0, 3 - len(f_run)), constant_values=np.nan))
    print(f"  run {seed+1}: {np.round(tracked[-1], 3)} Hz")

tracked = np.array(tracked)
print(f"Mode 1 across runs: mean={np.nanmean(tracked[:,0]):.3f} Hz, std={np.nanstd(tracked[:,0]):.3f} Hz "
      f"-> {'consistent' if np.nanstd(tracked[:,0]) < 0.1 else 'some run-to-run variability'}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

x = np.arange(min(3, len(freqs_ssi)))
axes[0].bar(x - 0.15, wn_true_hz[:len(x)], width=0.3, label="true", color="seagreen")
axes[0].bar(x + 0.15, freqs_ssi[:len(x)], width=0.3, label="SSI-COV", color="firebrick")
axes[0].set_xticks(x); axes[0].set_xticklabels([f"Mode {i+1}" for i in x])
axes[0].set_title("SSI-COV vs. true frequencies")
axes[0].set_ylabel("frequency [Hz]"); axes[0].legend(fontsize=8)

for i in range(min(3, tracked.shape[1])):
    axes[1].plot(range(1, 6), tracked[:, i], "o-", label=f"Mode {i+1}")
    axes[1].axhline(wn_true_hz[i] if i < len(wn_true_hz) else np.nan, color="gray", ls=":", lw=1)
axes[1].set_xlabel("independent ambient record #"); axes[1].set_ylabel("identified frequency [Hz]")
axes[1].set_title("Mode-tracking across 5 independent records")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA4_extension_ssi_cov.png"), dpi=150)
print("Saved outputs/moduleA4_extension_ssi_cov.png")
