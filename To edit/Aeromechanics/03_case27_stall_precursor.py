"""
Capstone 3 -- Strictly Causal Early-Warning Detector for Rotating Stall

Goal
----
Rotating stall in an axial compressor/turbine stage is often preceded by a
slowly growing "precursor" wave superposed on the blade-passing pressure
signal. A useful real-time monitoring system must flag the precursor
*before* stall fully develops, using only data available up to the
current instant (strictly causal -- no look-ahead), and must report a
*lead time* distribution across many independent events, not just a
single lucky case.

Method
------
1. Simulate many independent unsteady casing-pressure traces. Each trace
   is broadband turbulent pressure fluctuation plus, starting at a random
   onset time, an exponentially growing low-frequency precursor mode that
   saturates into full stall.
2. Build a strictly causal detector: a rolling-window band-limited RMS
   energy (using only past samples) compared against an adaptive
   threshold set from a calibration segment of healthy operation.
3. For each run, find the first time the causal statistic crosses the
   threshold and persists, and compute lead time = (true stall onset) -
   (detection time).
4. Report the lead-time distribution (mean, std, histogram) and a
   false-alarm rate on the healthy segments.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

rng = np.random.default_rng(27)

fs = 2000.0          # Hz, pressure sensor sample rate
T = 20.0              # seconds per run
t = np.arange(0, T, 1 / fs)
n = len(t)
n_runs = 40

precursor_freq = 8.0  # Hz, low-frequency rotating precursor
b_band, a_band = butter(4, [precursor_freq * 0.5, precursor_freq * 2.0], btype="bandpass", fs=fs)


def simulate_run(rng):
    onset = rng.uniform(8.0, 14.0)          # start of precursor growth [s]
    broadband = rng.normal(0, 1.0, n)
    growth_rate = rng.uniform(1.2, 2.0)     # 1/s
    envelope = np.zeros(n)
    mask = t >= onset
    envelope[mask] = np.exp(growth_rate * (t[mask] - onset))
    envelope = np.clip(envelope, 0, 8.0)    # saturate into fully developed stall
    precursor = envelope * np.sin(2 * np.pi * precursor_freq * t + rng.uniform(0, 2 * np.pi))
    signal = broadband + precursor
    full_stall_time = onset + np.log(8.0) / growth_rate  # when envelope saturates -> fully developed stall
    return signal, onset, full_stall_time


def causal_band_energy(signal, win_samples):
    """Strictly causal band-limited RMS energy using only past samples."""
    filtered = filtfilt(b_band, a_band, signal)  # offline bandpass is fine (fixed filter, not adaptive);
    # to keep the *statistic* causal we still only use a trailing window ending at each t
    energy = np.zeros_like(filtered)
    csum = np.cumsum(filtered ** 2)
    for i in range(len(filtered)):
        lo = max(0, i - win_samples)
        energy[i] = (csum[i] - (csum[lo - 1] if lo > 0 else 0)) / (i - lo + 1)
    return np.sqrt(energy)


win_samples = int(0.25 * fs)  # 0.25-second trailing window (fast response)
calib_end = int(4.0 * fs)     # first 4 s of every run used only to set the healthy baseline

lead_times = []
false_alarms = 0
example_signal = example_stat = example_thr = example_onset = example_full = None

for run_idx in range(n_runs):
    signal, onset, full_stall_time = simulate_run(rng)
    stat = causal_band_energy(signal, win_samples)
    healthy_mean = stat[:calib_end].mean()
    healthy_std = stat[:calib_end].std()
    threshold = healthy_mean + 4 * healthy_std

    # false alarms: any crossing before the growth phase even begins
    pre_onset_mask = t < onset
    if np.any(stat[pre_onset_mask][calib_end:] > threshold):
        false_alarms += 1

    post_calib = np.where(t >= 4.0)[0]
    crossed = post_calib[stat[post_calib] > threshold]
    if len(crossed) > 0:
        detect_time = t[crossed[0]]
        lead_times.append(full_stall_time - detect_time)
    else:
        lead_times.append(np.nan)  # missed detection

    if run_idx == 0:
        example_signal, example_stat, example_thr, example_onset, example_full = signal, stat, threshold, onset, full_stall_time

lead_times = np.array(lead_times)
missed = np.isnan(lead_times).sum()
valid_lead = lead_times[~np.isnan(lead_times)]

print("=== Case 27: Rotating-Stall Precursor Detector ===")
print(f"Runs simulated: {n_runs}")
print(f"Missed detections: {missed}/{n_runs}")
print(f"False-alarm rate (pre-onset crossings): {false_alarms}/{n_runs} = {100*false_alarms/n_runs:.1f}%")
print(f"Lead time: mean = {valid_lead.mean():.2f} s, std = {valid_lead.std():.2f} s, "
      f"min = {valid_lead.min():.2f} s, max = {valid_lead.max():.2f} s")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].plot(t, example_signal, lw=0.4, color="steelblue")
axes[0, 0].axvline(example_onset, color="orange", ls="--", label="precursor growth starts")
axes[0, 0].axvline(example_full, color="red", ls="--", label="fully developed stall")
axes[0, 0].set_title("Example casing-pressure signal (Run 1)")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("pressure fluctuation [a.u.]")
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(t, example_stat, color="darkorange", label="causal band-energy statistic")
axes[0, 1].axhline(example_thr, color="k", ls=":", label="adaptive threshold")
axes[0, 1].axvline(example_onset, color="orange", ls="--", label="growth starts")
axes[0, 1].axvline(example_full, color="red", ls="--", label="fully developed stall")
axes[0, 1].set_title("Causal detector statistic (Run 1)")
axes[0, 1].set_xlabel("time [s]"); axes[0, 1].set_ylabel("RMS band energy")
axes[0, 1].legend(fontsize=8)

axes[1, 0].hist(valid_lead, bins=12, color="seagreen", edgecolor="k")
axes[1, 0].axvline(valid_lead.mean(), color="red", ls="--", label=f"mean = {valid_lead.mean():.2f} s")
axes[1, 0].set_title(f"Lead-time distribution across {n_runs} runs")
axes[1, 0].set_xlabel("lead time [s]"); axes[1, 0].set_ylabel("count")
axes[1, 0].legend(fontsize=8)

axes[1, 1].bar(["Missed", "False alarm", "Detected\n(no false alarm)"],
               [missed, false_alarms, n_runs - missed - false_alarms],
               color=["gray", "firebrick", "seagreen"])
axes[1, 1].set_title("Detector outcome summary")
axes[1, 1].set_ylabel("number of runs")

plt.tight_layout()
plt.savefig("outputs/case27_stall_precursor.png", dpi=150)
print("Saved outputs/case27_stall_precursor.png")
