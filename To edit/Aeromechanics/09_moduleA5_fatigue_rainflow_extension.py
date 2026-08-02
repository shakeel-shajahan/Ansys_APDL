"""
Extension to Capstone 9 -- Paris-Law Crack Growth and Monte Carlo
Probabilistic Fatigue Life

Addresses reviewer feedback: "Paris law, fracture mechanics, crack
growth, probabilistic fatigue, Monte Carlo" beyond the base
rainflow+Miner (initiation-only, deterministic S-N) capstone.

Part A: Paris-law crack propagation. The base capstone's S-N/Miner
approach models fatigue CRACK INITIATION life. Once a small crack
initiates, its subsequent growth to critical size is governed by
fracture mechanics (Paris' law: da/dN = C*(Delta K)^m). We take the
rainflow cycle spectrum from the base capstone, convert each cycle's
bending-moment range to an equivalent stress-intensity-factor range via
a simplified single-edge-crack geometry factor, and integrate the Paris
law forward from an assumed initial flaw size to the critical crack size
(fracture toughness limit), giving a total propagation life in cycles.

Part B: Monte Carlo probabilistic fatigue. Real S-N and Paris-law
material constants have significant scatter. We propagate distributions
(not point values) for the S-N intercept, the Paris-law constants C and
m, and the initial flaw size through the full initiation + propagation
model, giving a fatigue-life PROBABILITY DISTRIBUTION rather than a
single deterministic number -- the standard way a reliability-based
certification actually reports blade life.
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

# ---------------------------------------------------------------
# Part A: Paris-law crack propagation for a single representative
# (deterministic) case, using the base capstone's rainflow cycle spectrum
# ---------------------------------------------------------------
# Simplified single-edge-notch geometry factor Y (dimensionless, ~1.12 for
# a shallow edge crack) converting nominal bending-moment range to a
# stress-intensity-factor range: Delta K = Y * Delta_sigma * sqrt(pi*a)
# We convert bending-moment range [kNm] to an equivalent nominal stress
# range [MPa] via an illustrative section modulus for the blade root.
section_modulus = 0.05  # m^3, illustrative blade-root section modulus
Y_geom = 1.12

def delta_K(range_kNm, a_m):
    delta_sigma_MPa = (range_kNm * 1e3) / section_modulus / 1e6  # kNm -> Nm -> Pa -> MPa
    return Y_geom * delta_sigma_MPa * np.sqrt(np.pi * a_m)  # MPa * sqrt(m)


C_paris = 6.9e-12   # Paris-law constant, typical for structural steel, units: m/cycle per (MPa*sqrt(m))^m
m_paris = 3.0        # Paris-law exponent, typical for steel
a0 = 0.2e-3          # initial flaw size [m], typical NDE detection threshold
K_IC = 60.0          # MPa*sqrt(m), fracture toughness (typical structural steel)
a_crit = min(0.02, (1 / np.pi) * (K_IC / (Y_geom * (ranges.max() * 1e3 / section_modulus / 1e6))) ** 2)

# Sort cycles by descending range (standard practice: apply the block of
# cycles from the rainflow count repeatedly until crack reaches a_crit)
cycle_ranges = ranges[counts > 0]
cycle_weights = counts[counts > 0]
order_idx = np.argsort(-cycle_ranges)
cycle_ranges = cycle_ranges[order_idx]
cycle_weights = cycle_weights[order_idx]

def propagate_crack(a_start, C, m, n_block_repeats=2000):
    """Block-stepped Paris-law integration: crack size is treated as constant
    within one 10-minute rainflow block (growth per block is tiny relative to
    a for these material constants), and the whole cycle spectrum's
    contribution is applied as one vectorised update per block -- the
    standard practical way to step Paris' law forward with a fixed load
    spectrum, instead of re-evaluating dK after every single cycle."""
    a = a_start
    cycles_per_block = cycle_weights.sum()
    n_cycles_total = 0.0
    for _ in range(n_block_repeats):
        if a >= a_crit:
            break
        dK_vec = delta_K(cycle_ranges, a)
        dK_vec = np.clip(dK_vec, 0, None)
        da_total = np.sum(C * dK_vec ** m * cycle_weights)
        a += da_total
        n_cycles_total += cycles_per_block
    return n_cycles_total


n_cycles_to_failure = propagate_crack(a0, C_paris, m_paris)
block_seconds = T  # from base capstone, T = 600 s (10-minute block)
cycles_per_block = counts.sum()
blocks_to_failure = n_cycles_to_failure / cycles_per_block
propagation_life_hours = blocks_to_failure * block_seconds / 3600.0

print("=== Extension 9a: Paris-Law Crack Propagation ===")
print(f"Critical crack size a_crit = {a_crit*1000:.2f} mm (fracture toughness limit)")
print(f"Initial flaw size a0       = {a0*1000:.2f} mm (NDE detection threshold)")
print(f"Cycles to propagate from a0 to a_crit under this duty cycle: {n_cycles_to_failure:.0f}")
print(f"Equivalent propagation life: {propagation_life_hours:.1f} hours "
      f"({propagation_life_hours/24:.1f} days) of this specific representative load case repeated")
print("(Recall: this is PROPAGATION life only, starting from an already-initiated flaw --")
print(" it is additive to, not a replacement for, the initiation life from the base capstone's")
print(" Miner's-rule calculation.)")

# ---------------------------------------------------------------
# Part B: Monte Carlo probabilistic fatigue -- propagate scatter in the
# S-N intercept, Paris constants C, and initial flaw size a0
# ---------------------------------------------------------------
rng_mc = np.random.default_rng(905)
n_mc = 500

# Log-normal scatter, typical coefficients of variation for these quantities
a_coef_samples = a_coef * 10 ** rng_mc.normal(0, 0.15, n_mc)     # ~35% CoV on S-N intercept (log-scale)
C_samples = C_paris * 10 ** rng_mc.normal(0, 0.20, n_mc)          # Paris C scatter (log-normal, typical)
a0_samples = np.clip(rng_mc.normal(a0, 0.05e-3, n_mc), 0.05e-3, None)

initiation_damage_per_block = np.sum(counts / (a_coef_samples[:, None] * S_eq[None, :] ** (-m_exp)), axis=1)
initiation_life_blocks = 1.0 / initiation_damage_per_block

propagation_cycles_mc = np.array([
    propagate_crack(a0_samples[i], C_samples[i], m_paris, n_block_repeats=500)
    for i in range(n_mc)
])
propagation_life_blocks_mc = propagation_cycles_mc / cycles_per_block

total_life_hours_mc = (initiation_life_blocks + propagation_life_blocks_mc) * block_seconds / 3600.0
total_life_days_mc = total_life_hours_mc / 24.0

p10, p50, p90 = np.percentile(total_life_days_mc, [10, 50, 90])
print("\n=== Extension 9b: Monte Carlo Probabilistic Fatigue Life ===")
print(f"Monte Carlo runs: {n_mc}")
print(f"Total life (initiation + propagation), this load case repeated:")
print(f"  P10 (10th percentile, conservative design life): {p10:.2f} days")
print(f"  P50 (median)                                    : {p50:.2f} days")
print(f"  P90 (90th percentile, optimistic)                : {p90:.2f} days")
print(f"Spread P90/P10 = {p90/p10:.1f}x -- illustrates why a single deterministic life number")
print("(as in the base capstone) hides substantial material/manufacturing scatter that a")
print("reliability-based design would explicitly carry forward into a safety factor.")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

a_history = [a0]
a = a0
for _ in range(200):
    if a >= a_crit:
        break
    dK_vec = np.clip(delta_K(cycle_ranges, a), 0, None)
    a += np.sum(C_paris * dK_vec ** m_paris * cycle_weights)
    a_history.append(a)
axes[0].plot(a_history, color="firebrick")
axes[0].axhline(a_crit, color="k", ls="--", label=f"critical size a_crit={a_crit*1000:.1f} mm")
axes[0].set_xlabel("load-block repeats"); axes[0].set_ylabel("crack size a [m]")
axes[0].set_title("Paris-law crack growth trajectory")
axes[0].legend(fontsize=8)

axes[1].hist(total_life_days_mc, bins=40, color="steelblue", edgecolor="k")
axes[1].axvline(p10, color="firebrick", ls="--", label=f"P10={p10:.1f} d")
axes[1].axvline(p50, color="black", ls="-", label=f"P50={p50:.1f} d")
axes[1].axvline(p90, color="seagreen", ls="--", label=f"P90={p90:.1f} d")
axes[1].set_xlabel("total fatigue life [days, this load case repeated]"); axes[1].set_ylabel("count")
axes[1].set_title(f"Monte Carlo probabilistic life distribution (n={n_mc})")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA5_extension_paris_monte_carlo.png"), dpi=150)
print("Saved outputs/moduleA5_extension_paris_monte_carlo.png")
