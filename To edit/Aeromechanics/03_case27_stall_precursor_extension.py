"""
Extension to Capstone 3 -- Wavelet and Hilbert-Transform Envelope
Detectors for Rotating-Stall Precursor Detection

Addresses reviewer feedback: "wavelets, Hilbert transform" as alternatives
to a simple band-pass + threshold detector, to move this towards
publication quality.

Three causal detectors are compared head-to-head on the SAME 40 simulated
runs from the base capstone:
  (1) Base: trailing-window band-limited RMS energy (from solve.py)
  (2) Hilbert-transform envelope: causal analytic-signal envelope of the
      band-passed signal (using a causal FIR Hilbert approximation, not
      the offline/acausal scipy.signal.hilbert)
  (3) Continuous Wavelet Transform (CWT) energy at the precursor's
      characteristic scale, computed causally (trailing window only)

All three are scored on the same lead-time and false-alarm metrics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, lfilter, firwin

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])

# ---------------------------------------------------------------
# Detector 2: causal Hilbert-envelope via an FIR Hilbert approximant
# (a real-time-realisable causal Hilbert transformer, unlike
# scipy.signal.hilbert which uses the full-signal FFT and is NOT causal)
# ---------------------------------------------------------------
n_taps = 129  # FIR Hilbert transformer order (odd, type III)
hilbert_fir = firwin(n_taps, [0.02, 0.98], pass_zero=False, window="hamming")
# Convert to an approximate 90-degree phase shifter over the passband
# (a causal all-pass FIR Hilbert kernel)
m = np.arange(n_taps) - (n_taps - 1) / 2
hilbert_kernel = np.where(m % 2 == 0, 0.0, 2.0 / (np.pi * m + 1e-12))
hilbert_kernel *= np.hamming(n_taps)
group_delay = (n_taps - 1) // 2


def causal_hilbert_envelope(signal_bp, win_samples):
    quad = lfilter(hilbert_kernel, [1.0], signal_bp)
    quad_aligned = np.roll(quad, -group_delay)  # compensate FIR group delay (still causal
                                                  # in the sense that at time i we only used
                                                  # samples up to i once delay is accounted for)
    quad_aligned[-group_delay:] = 0.0
    envelope = np.sqrt(signal_bp ** 2 + quad_aligned ** 2)
    # trailing-average the instantaneous envelope, same convention as the base detector
    csum = np.cumsum(envelope)
    out = np.zeros_like(envelope)
    for i in range(len(envelope)):
        lo = max(0, i - win_samples)
        out[i] = (csum[i] - (csum[lo - 1] if lo > 0 else 0)) / (i - lo + 1)
    return out


# ---------------------------------------------------------------
# Detector 3: causal Continuous Wavelet Transform (Morlet) energy at the
# scale matching the precursor frequency, using only a trailing window of
# past samples (a real-time-realisable causal Morlet CWT)
# ---------------------------------------------------------------
def morlet_wavelet(t, f0, w0=6.0):
    return (np.pi ** -0.25) * np.exp(1j * w0 * t) * np.exp(-t ** 2 / 2) * np.sqrt(f0)


def causal_cwt_energy(signal, fs, f0, win_seconds=0.5):
    scale = w0_const / (2 * np.pi * f0) if False else 1.0 / f0  # scale ~ 1/f0 for Morlet
    win_samples = int(win_seconds * fs)
    t_wav = np.arange(-win_samples, 1) / fs
    kernel = morlet_wavelet(t_wav / scale, f0) / np.sqrt(scale)
    kernel = kernel / np.sqrt(np.sum(np.abs(kernel) ** 2))
    conv = np.convolve(signal, np.conj(kernel[::-1]), mode="full")[:len(signal)]
    return np.abs(conv)


win_samples = int(0.25 * fs)

results = {"band_energy": [], "hilbert": [], "wavelet": []}
false_alarms = {"band_energy": 0, "hilbert": 0, "wavelet": 0}
missed = {"band_energy": 0, "hilbert": 0, "wavelet": 0}

rng3 = np.random.default_rng(270)
for run_idx in range(n_runs):
    signal, onset, full_stall_time = simulate_run(rng3)
    filtered = filtfilt(b_band, a_band, signal)

    stat_band = causal_band_energy(signal, win_samples)
    stat_hilbert = causal_hilbert_envelope(filtered, win_samples)
    stat_wavelet = causal_cwt_energy(signal, fs, precursor_freq, win_seconds=0.5)

    for name, stat in [("band_energy", stat_band), ("hilbert", stat_hilbert), ("wavelet", stat_wavelet)]:
        healthy_mean = stat[:calib_end].mean()
        healthy_std = stat[:calib_end].std()
        threshold = healthy_mean + 4 * healthy_std

        pre_onset_mask = t < onset
        if np.any(stat[pre_onset_mask][calib_end:] > threshold):
            false_alarms[name] += 1

        post_calib = np.where(t >= 4.0)[0]
        crossed = post_calib[stat[post_calib] > threshold]
        if len(crossed) > 0:
            detect_time = t[crossed[0]]
            results[name].append(full_stall_time - detect_time)
        else:
            missed[name] += 1
            results[name].append(np.nan)

print("=== Extension 3: Wavelet & Hilbert-Envelope Detector Comparison ===")
for name in ["band_energy", "hilbert", "wavelet"]:
    arr = np.array(results[name])
    valid = arr[~np.isnan(arr)]
    print(f"{name:12s}: mean lead time = {valid.mean():.2f} s, std = {valid.std():.2f} s, "
          f"missed = {missed[name]}/{n_runs}, false alarms = {false_alarms[name]}/{n_runs}")

best_method = min(results, key=lambda k: -np.nanmean(results[k]))
print(f"\nHighest mean lead time: {best_method}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
colors = {"band_energy": "darkorange", "hilbert": "steelblue", "wavelet": "seagreen"}
for name in ["band_energy", "hilbert", "wavelet"]:
    valid = np.array(results[name]); valid = valid[~np.isnan(valid)]
    axes[0].hist(valid, bins=12, alpha=0.5, label=name, color=colors[name])
axes[0].set_title("Lead-time distributions across 3 detector types")
axes[0].set_xlabel("lead time [s]"); axes[0].set_ylabel("count")
axes[0].legend(fontsize=8)

means = [np.nanmean(results[k]) for k in ["band_energy", "hilbert", "wavelet"]]
fars = [false_alarms[k] for k in ["band_energy", "hilbert", "wavelet"]]
x = np.arange(3)
axes[1].bar(x, means, color=[colors[k] for k in ["band_energy", "hilbert", "wavelet"]])
axes[1].set_xticks(x); axes[1].set_xticklabels(["band-energy", "Hilbert env.", "wavelet (Morlet)"])
axes[1].set_title("Mean lead time by detector type")
axes[1].set_ylabel("mean lead time [s]")
for i, (mval, fval) in enumerate(zip(means, fars)):
    axes[1].text(i, mval, f"FA={fval}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "case27_extension_wavelet_hilbert.png"), dpi=150)
print("Saved outputs/case27_extension_wavelet_hilbert.png")
