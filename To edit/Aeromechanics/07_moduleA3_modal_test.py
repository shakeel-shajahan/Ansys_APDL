"""
Capstone 7 -- Low-Cost Cantilever-Beam Modal Test vs. Euler-Bernoulli
Beam Theory

Goal
----
Simulate a simple, low-cost impulse-hammer modal test on a cantilever
beam (e.g. a scaled stand-in for a turbine blade last-stage sector before
a full 3D FE model is built), identify the first three bending modes from
the simulated response, and validate the identified frequencies against
closed-form Euler-Bernoulli cantilever theory.

Method
------
1. Euler-Bernoulli cantilever beam: closed-form natural frequencies
   f_n = (beta_n L)^2 / (2 pi L^2) * sqrt(EI / (rho A)).
2. Build a reduced modal model (first 4 modes) and simulate the transient
   response to an impulsive hammer strike near the tip, with output
   measured at the tip (typical single-accelerometer roving-hammer test).
3. Estimate the FRF via FFT of impulse response (impact test convention:
   H(f) = X(f) / F(f)), identify peaks -> natural frequencies.
4. Compare identified vs. closed-form theoretical frequencies.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ---------------------------------------------------------------
# 1. Cantilever beam properties (aluminium-like flat bar, illustrative)
# ---------------------------------------------------------------
E = 70e9        # Pa
rho = 2700.0    # kg/m^3
L = 0.30        # m
width = 0.02    # m
thick = 0.003   # m
A = width * thick
I = width * thick ** 3 / 12
EI = E * I
mA = rho * A

beta_n_L = np.array([1.8751, 4.6941, 7.8548, 10.9955])  # first four cantilever eigenvalues
f_theory = (beta_n_L ** 2 / (2 * np.pi * L ** 2)) * np.sqrt(EI / mA)
zeta_assumed = np.array([0.008, 0.006, 0.005, 0.004])  # slight increase in damping for higher modes (typical)

print("=== Module A3: Cantilever Beam Modal Test ===")
print(f"Beam: L={L*1000:.0f} mm, {width*1000:.0f}x{thick*1000:.1f} mm section, E={E/1e9:.0f} GPa, rho={rho} kg/m^3")
print(f"Theoretical (Euler-Bernoulli) frequencies [Hz]: {np.round(f_theory, 2)}")

# ---------------------------------------------------------------
# 2. Modal superposition response to an impulsive hammer hit near the tip,
#    measured at the tip -- both hammer and sensor locations weighted by
#    the (normalized) mode shape value there.
# ---------------------------------------------------------------
def mode_shape_tip_factor(n):
    """Simplified relative modal participation factor at/near the free tip (illustrative, not exact)."""
    return 1.0 - 0.05 * n  # near tip, all modes contribute strongly; slight roll-off assumed for realism


fs = 5000.0
T = 4.0
t = np.arange(0, T, 1 / fs)
response = np.zeros_like(t)
rng = np.random.default_rng(3)
impulse_force = 1.0  # N, hammer impulse (delta-like, applied at t=0)

for n in range(4):
    wn = 2 * np.pi * f_theory[n]
    zeta = zeta_assumed[n]
    wd = wn * np.sqrt(1 - zeta ** 2)
    phi = mode_shape_tip_factor(n) ** 2  # hammer-at-tip * response-at-tip
    modal_mass = 1.0  # normalized
    resp_n = (impulse_force * phi / (modal_mass * wd)) * np.exp(-zeta * wn * t) * np.sin(wd * t)
    response += resp_n

response_meas = response + rng.normal(0, response.std() * 0.03, len(t))
force_signal = np.zeros_like(t)
force_signal[0] = impulse_force * fs  # discrete-time impulse approximation

# ---------------------------------------------------------------
# 3. FRF via FFT (impact test): H(f) = FFT(response) / FFT(force)
# ---------------------------------------------------------------
N = len(t)
window = np.ones(N)
Xf = np.fft.rfft(response_meas * window)
Ff = np.fft.rfft(force_signal * window)
freqs = np.fft.rfftfreq(N, d=1 / fs)
H = Xf / (Ff + 1e-12)
mag = np.abs(H)

mask = (freqs > 5) & (freqs < 1000)
peaks, _ = find_peaks(mag[mask], prominence=np.max(mag[mask]) * 0.003)
f_identified = freqs[mask][peaks]

print(f"Identified peaks from simulated impact test [Hz]: {np.round(f_identified[:4], 2)}")
err = 100 * (f_identified[:4] - f_theory[:len(f_identified[:4])]) / f_theory[:len(f_identified[:4])]
print(f"Relative error vs. theory [%]: {np.round(err, 2)}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].plot(t[:1000], response[:1000], color="steelblue")
axes[0, 0].set_title("Simulated tip response to hammer impulse (first 0.2 s)")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("displacement [a.u.]")

axes[0, 1].semilogy(freqs[mask], mag[mask], color="darkorange")
for f in f_identified[:4]:
    axes[0, 1].axvline(f, color="red", ls="--", lw=0.7)
axes[0, 1].set_title("FRF magnitude (impact test) with identified peaks")
axes[0, 1].set_xlabel("frequency [Hz]"); axes[0, 1].set_ylabel("|H(f)|")

x = np.arange(len(f_theory))
axes[1, 0].bar(x - 0.15, f_theory, width=0.3, color="seagreen", label="Euler-Bernoulli theory")
axes[1, 0].bar(x[:len(f_identified[:4])] + 0.15, f_identified[:4], width=0.3, color="firebrick", label="identified (test)")
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels([f"Mode {i+1}" for i in x])
axes[1, 0].set_title("Theory vs. simulated-test frequencies")
axes[1, 0].set_ylabel("frequency [Hz]"); axes[1, 0].legend(fontsize=8)

xf = np.linspace(0, L, 200)
def mode_shape(n, x):
    beta = beta_n_L[n] / L
    sigma = (np.cosh(beta_n_L[n]) + np.cos(beta_n_L[n])) / (np.sinh(beta_n_L[n]) + np.sin(beta_n_L[n]))
    return (np.cosh(beta * x) - np.cos(beta * x)) - sigma * (np.sinh(beta * x) - np.sin(beta * x))

for n in range(3):
    shape = mode_shape(n, xf)
    shape /= np.max(np.abs(shape))
    axes[1, 1].plot(xf, shape + 2 * n, label=f"Mode {n+1} ({f_theory[n]:.1f} Hz)")
axes[1, 1].set_title("First three cantilever mode shapes (theory)")
axes[1, 1].set_xlabel("position along beam [m]"); axes[1, 1].set_yticks([])
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/moduleA3_modal_test.png", dpi=150)
print("Saved outputs/moduleA3_modal_test.png")
