"""
Capstone 5 -- Frequency Response Function (FRF) Estimation and Modal
Parameter Identification from Broadband Time Histories

Goal
----
Estimate the FRF of a lightly damped 3-DOF structure (a coarse stand-in
for a blade-disc sector or a test rig bracket) from simulated force and
response time histories under band-limited random excitation, using the
H1 estimator, and identify natural frequencies and modal damping ratios
via the half-power (3 dB) bandwidth method -- the standard first step of
any experimental modal test before a full curve fit.

Method
------
1. Build a 3-DOF mass-spring-damper chain, simulate its response to
   white-noise base/force excitation using state-space time integration.
2. Add measurement noise on both force and response channels.
3. Estimate the FRF via Welch's method: H1 = Sxy / Sxx (cross-spectrum
   over input auto-spectrum), plus coherence.
4. Identify each resonance peak, then estimate damping ratio from the
   half-power bandwidth around each peak.
5. Compare identified frequencies/damping against the true eigenvalues of
   the system matrices.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import csd, welch, coherence
from scipy.integrate import solve_ivp
from scipy.linalg import eig

rng = np.random.default_rng(1)

# ---------------------------------------------------------------
# 1. 3-DOF mass-spring-damper chain (fixed-free)
# ---------------------------------------------------------------
m = np.array([1.0, 1.2, 0.8])
k = np.array([4000.0, 3000.0, 2000.0])   # k1 (ground-m1), k2 (m1-m2), k3 (m2-m3)
c = 0.002 * k  # light proportional-ish damping

M = np.diag(m)
K = np.array([[k[0] + k[1], -k[1], 0],
              [-k[1], k[1] + k[2], -k[2]],
              [0, -k[2], k[2]]])
C = np.array([[c[0] + c[1], -c[1], 0],
              [-c[1], c[1] + c[2], -c[2]],
              [0, -c[2], c[2]]])

# true modal properties
evals, evecs = eig(K, M)
wn_true = np.sqrt(np.real(evals))
order = np.argsort(wn_true)
wn_true = wn_true[order] / (2 * np.pi)  # Hz
# approximate modal damping ratios via generalized modal damping projection
Phi = np.real(evecs[:, order])
Mm = Phi.T @ M @ Phi
Cm = Phi.T @ C @ Phi
Km = Phi.T @ K @ Phi
zeta_true = np.diag(Cm) / (2 * np.sqrt(np.diag(Mm) * np.diag(Km)))

# ---------------------------------------------------------------
# 2. Time-domain simulation with broadband force at DOF 1
# ---------------------------------------------------------------
fs = 2000.0
T = 60.0
t = np.arange(0, T, 1 / fs)
n = len(t)
force = rng.normal(0, 1.0, n)  # broadband white-noise force on m1
force_interp_dt = 1 / fs

Minv = np.linalg.inv(M)


def rhs(ti, y):
    x = y[:3]; v = y[3:]
    idx = min(int(ti / force_interp_dt), n - 1)
    f = np.array([force[idx], 0, 0])
    a = Minv @ (f - C @ v - K @ x)
    return np.concatenate([v, a])


sol = solve_ivp(rhs, [0, T], np.zeros(6), t_eval=t, max_step=1 / fs)
x1 = sol.y[0]
x2 = sol.y[1]
x3 = sol.y[2]

meas_noise = 0.02
x1_meas = x1 + rng.normal(0, meas_noise * x1.std(), n)
force_meas = force + rng.normal(0, meas_noise * force.std(), n)

# ---------------------------------------------------------------
# 3. H1 FRF estimator: H1(f) = Sxy(f) / Sxx(f)   (Sxy = force-response cross spectrum)
# ---------------------------------------------------------------
nperseg = 4096
f_csd, Sfx = csd(force_meas, x1_meas, fs=fs, nperseg=nperseg)
_, Sff = welch(force_meas, fs=fs, nperseg=nperseg)
H1 = Sfx / Sff
f_coh, gamma2 = coherence(force_meas, x1_meas, fs=fs, nperseg=nperseg)

mag = np.abs(H1)
mask = (f_csd > 2) & (f_csd < 200)

# ---------------------------------------------------------------
# 4. Peak-picking + half-power bandwidth damping estimate
# ---------------------------------------------------------------
from scipy.signal import find_peaks
peaks, _ = find_peaks(mag[mask], prominence=np.max(mag[mask]) * 0.05)
f_sub = f_csd[mask]; mag_sub = mag[mask]
identified = []
for p in peaks:
    fn = f_sub[p]
    peak_val = mag_sub[p]
    half_power = peak_val / np.sqrt(2)
    # search left/right for half-power crossing
    left = p
    while left > 0 and mag_sub[left] > half_power:
        left -= 1
    right = p
    while right < len(mag_sub) - 1 and mag_sub[right] > half_power:
        right += 1
    if right > left:
        f1, f2 = f_sub[left], f_sub[right]
        zeta_est = (f2 - f1) / (2 * fn)
        identified.append((fn, zeta_est))

print("=== Module A1: FRF Estimation & Modal Identification ===")
print(f"True modal frequencies [Hz]: {np.round(wn_true, 2)}")
print(f"True modal damping ratios  : {np.round(zeta_true, 4)}")
print("Identified from H1 FRF (peak-picking + half-power bandwidth):")
for fn, z in identified:
    print(f"  f_n = {fn:6.2f} Hz   zeta = {z:.4f}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].plot(t[:2000], x1[:2000], lw=0.6)
axes[0, 0].set_title("Simulated response time history at DOF 1 (first 1 s)")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("displacement [a.u.]")

axes[0, 1].semilogy(f_csd[mask], mag[mask], color="steelblue")
for fn, z in identified:
    axes[0, 1].axvline(fn, color="red", ls="--", lw=0.8)
axes[0, 1].set_title("H1 FRF magnitude with identified resonances")
axes[0, 1].set_xlabel("frequency [Hz]"); axes[0, 1].set_ylabel("|H1(f)|")

axes[1, 0].plot(f_coh[mask], gamma2[mask], color="darkorange")
axes[1, 0].set_title("Coherence (data quality check)")
axes[1, 0].set_xlabel("frequency [Hz]"); axes[1, 0].set_ylabel(r"$\gamma^2$")
axes[1, 0].set_ylim(0, 1.05)

ids = np.array([f for f, _ in identified][:3])
true3 = wn_true[:len(ids)]
axes[1, 1].bar(np.arange(len(ids)) - 0.15, true3, width=0.3, label="true", color="seagreen")
axes[1, 1].bar(np.arange(len(ids)) + 0.15, ids, width=0.3, label="identified", color="firebrick")
axes[1, 1].set_xticks(np.arange(len(ids)))
axes[1, 1].set_xticklabels([f"Mode {i+1}" for i in range(len(ids))])
axes[1, 1].set_title("True vs. identified natural frequencies")
axes[1, 1].set_ylabel("frequency [Hz]")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/moduleA1_frf.png", dpi=150)
print("Saved outputs/moduleA1_frf.png")
