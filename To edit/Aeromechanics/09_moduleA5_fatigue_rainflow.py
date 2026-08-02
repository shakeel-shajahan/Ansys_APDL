"""
Capstone 9 -- Aeroelastic Load Simulation, Rainflow Counting, and Fatigue
Damage Accumulation (Miner's Rule)

Goal
----
A turbine blade root sees a fluctuating bending-moment time series driven
by turbulent inflow and rotational once-per-rev loading. This capstone
simulates a representative blade-root bending-moment signal, applies the
industry-standard rainflow-counting algorithm to extract closed stress
cycles, and estimates fatigue damage accumulation using an S-N curve and
Miner's linear damage rule -- exactly the workflow used to certify blade
life against a design duty cycle.

Method
------
1. Simulate a 10-minute blade-root bending-moment time series: a 1P
   (once-per-rev) periodic component + turbulence-driven broadband
   fluctuation + a slow gust-induced envelope modulation.
2. Rainflow-count the signal (ASTM E1049-85 simplified 4-point algorithm)
   to get cycle (range, mean) pairs.
3. Apply a Goodman mean-stress correction, then an S-N curve
   (log(N) = log(a) - m*log(S)) to get cycles-to-failure per bin.
4. Miner's rule: D = sum(n_i / N_i); estimate remaining life at this duty
   cycle.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)

# ---------------------------------------------------------------
# 1. Simulate blade-root bending-moment time series
# ---------------------------------------------------------------
fs = 50.0
T = 600.0  # 10 minutes, a standard turbulence simulation block length
t = np.arange(0, T, 1 / fs)
n = len(t)

rotor_speed_hz = 0.2  # 12 rpm -> 1P frequency
M_mean = 800.0        # kNm, mean bending moment
M_1P = 250.0 * np.sin(2 * np.pi * rotor_speed_hz * t)
M_3P = 60.0 * np.sin(2 * np.pi * 3 * rotor_speed_hz * t + 0.4)

# turbulent component: filtered white noise (simple AR(1) as a von-Karman-like proxy)
turb = np.zeros(n)
alpha = 0.995
white = rng.normal(0, 1.0, n)
for i in range(1, n):
    turb[i] = alpha * turb[i - 1] + np.sqrt(1 - alpha ** 2) * white[i]
turb *= 180.0  # kNm scale

gust_envelope = 1.0 + 0.3 * np.sin(2 * np.pi * (1 / 120) * t)  # slow gust modulation, 2-min period

M = M_mean + gust_envelope * (M_1P + M_3P + turb)

print("=== Module A5: Aeroelastic Fatigue via Rainflow Counting ===")
print(f"Simulated {T/60:.0f}-minute blade-root bending-moment signal, "
      f"mean={M.mean():.1f} kNm, std={M.std():.1f} kNm, "
      f"max={M.max():.1f} kNm, min={M.min():.1f} kNm")


# ---------------------------------------------------------------
# 2. Rainflow counting (ASTM E1049-85 simplified 4-point algorithm)
# ---------------------------------------------------------------
def find_reversals(signal):
    """Reduce a signal to its turning points (local extrema)."""
    d = np.diff(signal)
    d[d == 0] = 1e-12
    sign = np.sign(d)
    idx = np.where(np.diff(sign) != 0)[0] + 1
    return np.concatenate([[0], idx, [len(signal) - 1]])


def rainflow_counting(signal):
    """Standard 4-point rainflow counting. Returns list of (range, mean, count) with
    count=1.0 for full cycles and 0.5 for the half-cycles left in the residual."""
    turning_idx = find_reversals(signal)
    points = signal[turning_idx].tolist()
    cycles = []
    stack = []
    for x in points:
        stack.append(x)
        while len(stack) >= 4:
            x1, x2, x3, x4 = stack[-4], stack[-3], stack[-2], stack[-1]
            r1 = abs(x2 - x1)
            r2 = abs(x3 - x2)
            r3 = abs(x4 - x3)
            if r2 <= r1 and r2 <= r3:
                rng_ = r2
                mean_ = (x2 + x3) / 2
                cycles.append((rng_, mean_, 1.0))
                del stack[-3:-1]
            else:
                break
    # residual half-cycles
    for i in range(len(stack) - 1):
        r = abs(stack[i + 1] - stack[i])
        m = (stack[i + 1] + stack[i]) / 2
        cycles.append((r, m, 0.5))
    return cycles


cycles = rainflow_counting(M)
ranges = np.array([c[0] for c in cycles])
means = np.array([c[1] for c in cycles])
counts = np.array([c[2] for c in cycles])

print(f"Rainflow cycles extracted: {len(cycles)} "
      f"({np.sum(counts==1.0):.0f} full, {np.sum(counts==0.5)*2:.0f} half-cycles combined)")
print(f"Cycle range: min={ranges.min():.1f} kNm, max={ranges.max():.1f} kNm, "
      f"mean={ranges.mean():.1f} kNm")

# ---------------------------------------------------------------
# 3. Goodman mean-stress correction + S-N curve + Miner's rule
# ---------------------------------------------------------------
UTS = 4000.0  # kNm-equivalent "ultimate" root capacity (illustrative)
S_eq = ranges / (1 - means / UTS)  # Goodman-corrected equivalent fully-reversed range

# S-N curve: N = a * S^-m, calibrated from one reference coupon-test point,
# typical of a steel / Ti-alloy last-stage blade root (moderate HCF slope, m ~ 4.5)
m_exp = 4.5
S_ref = 900.0   # kNm, reference fully-reversed range from a coupon/sub-component fatigue test
N_ref = 1.0e5   # cycles to failure at S_ref (typical metallic HCF reference point)
a_coef = N_ref * S_ref ** m_exp

N_fail = a_coef * S_eq ** (-m_exp)
damage_per_cycle = counts / N_fail
total_damage_per_block = damage_per_cycle.sum()

seconds_per_year = 8760.0 * 3600.0
blocks_per_year = seconds_per_year / T
annual_damage = total_damage_per_block * blocks_per_year
life_years = 1.0 / annual_damage if annual_damage > 0 else np.inf

print(f"Damage accumulated in this single {T/60:.0f}-minute load case: {total_damage_per_block:.3e}")
print(f"If this exact load case repeated continuously (a diagnostic upper-bound, NOT a life claim): "
      f"{life_years*365.25:.2f} days equivalent")
print("Interpretation: this representative block was drawn from a single, comparatively severe")
print("turbulence realisation. A certified fatigue-life estimate per IEC 61400-1 / equivalent")
print("turbomachinery practice sums Miner damage across the FULL duty-cycle histogram (many load")
print("cases across the operating/wind-speed distribution, each weighted by its probability of")
print("occurrence) -- never extrapolates one worst-case block. This capstone demonstrates the")
print("per-load-case rainflow + Miner mechanics; the project task below asks you to complete the")
print("full multi-load-case summation.")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

axes[0, 0].plot(t, M, lw=0.4, color="steelblue")
axes[0, 0].set_title("Simulated blade-root bending-moment time series")
axes[0, 0].set_xlabel("time [s]"); axes[0, 0].set_ylabel("bending moment [kNm]")

sc = axes[0, 1].scatter(means, ranges, c=counts, cmap="coolwarm", s=8, alpha=0.7)
axes[0, 1].set_title("Rainflow cycle scatter (range vs. mean)")
axes[0, 1].set_xlabel("cycle mean [kNm]"); axes[0, 1].set_ylabel("cycle range [kNm]")
plt.colorbar(sc, ax=axes[0, 1], label="cycle count (0.5 or 1.0)")

axes[1, 0].hist(ranges, bins=30, weights=counts, color="seagreen", edgecolor="k")
axes[1, 0].set_title("Rainflow range histogram (cycle-count weighted)")
axes[1, 0].set_xlabel("cycle range [kNm]"); axes[1, 0].set_ylabel("weighted count")

S_curve = np.linspace(S_eq.min() * 0.8, S_eq.max() * 1.2, 200)
N_curve = a_coef * S_curve ** (-m_exp)
axes[1, 1].loglog(N_curve, S_curve, color="black", label="S-N curve")
axes[1, 1].scatter(N_fail, S_eq, c=damage_per_cycle, cmap="Reds", s=10, label="rainflow cycles")
axes[1, 1].set_title("S-N curve with rainflow cycles overlaid")
axes[1, 1].set_xlabel("cycles to failure, N"); axes[1, 1].set_ylabel("Goodman-corrected range, S [kNm]")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/moduleA5_fatigue_rainflow.png", dpi=150)
print("Saved outputs/moduleA5_fatigue_rainflow.png")
