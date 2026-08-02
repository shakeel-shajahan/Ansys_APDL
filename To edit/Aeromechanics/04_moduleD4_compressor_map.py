"""
Capstone 4 -- Compressor/Turbine Map Reconstruction and Surge/Choke
Margin Quantification

Goal
----
In service, only a sparse set of operating points is ever instrumented on
a real machine (a handful of speed lines, a few points per line). This
capstone reconstructs a full performance map from such sparse data using
2D thin-plate-spline-style radial-basis interpolation, and then computes
an operability margin (surge margin analogue) as a function of speed,
along with a bootstrap-based uncertainty band on that margin -- directly
relevant to assessing the safe operating envelope of an LP last stage
across the turbine's speed range.

Method
------
1. Ground-truth compressor map (pressure ratio vs. corrected flow, per
   speed line) with a surge line defined as the flow at which dPR/dmdot
   changes sign (peak of each speed line).
2. Sparse "measured" data: 5 speed lines x 6 points each = 30 points.
3. Reconstruct full map via RBF interpolation (thin-plate spline kernel).
4. Extract surge line and choke line from the reconstructed map, compute
   surge margin = (mdot_operating - mdot_surge) / mdot_surge for an
   assumed operating line.
5. Bootstrap over the 30 measured points (resample with replacement,
   refit) to get a confidence band on the surge-margin curve.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

rng = np.random.default_rng(4)


def true_map(mdot, speed):
    center = 0.45 + 0.5 * speed
    pr = 1.1 + 0.9 * speed ** 1.6 - 3.2 * (mdot - center) ** 2 / (0.15 + 0.3 * speed)
    return np.clip(pr, 1.0, None)


speed_lines = np.array([0.7, 0.8, 0.9, 1.0, 1.05])
n_per_line = 6
mdot_meas, speed_meas, pr_meas = [], [], []
for s in speed_lines:
    center = 0.45 + 0.5 * s
    m = np.linspace(center - 0.16, center + 0.10, n_per_line)
    p = true_map(m, s) + rng.normal(0, 0.01, n_per_line)
    mdot_meas.append(m); speed_meas.append(np.full(n_per_line, s)); pr_meas.append(p)
mdot_meas = np.concatenate(mdot_meas)
speed_meas = np.concatenate(speed_meas)
pr_meas = np.concatenate(pr_meas)

X_meas = np.column_stack([mdot_meas, speed_meas])


def fit_and_reconstruct(X, y, mdot_grid, speed_grid):
    rbf = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=1e-3)
    MM, SS = np.meshgrid(mdot_grid, speed_grid)
    Xg = np.column_stack([MM.ravel(), SS.ravel()])
    pr = rbf(Xg).reshape(MM.shape)
    return MM, SS, pr


mdot_grid = np.linspace(0.35, 1.05, 120)
speed_grid_fine = np.linspace(0.68, 1.06, 80)
MM, SS, PR = fit_and_reconstruct(X_meas, pr_meas, mdot_grid, speed_grid_fine)


def surge_flow_for_speed(mdot_grid, pr_row):
    """Surge point = peak of the speed-line curve (dPR/dmdot = 0, before the unstable branch)."""
    idx = np.argmax(pr_row)
    return mdot_grid[idx]


surge_line = np.array([surge_flow_for_speed(mdot_grid, PR[i, :]) for i in range(len(speed_grid_fine))])
operating_line = 0.6 + 0.28 * speed_grid_fine  # assumed nominal operating line (design choice)
surge_margin = (operating_line - surge_line) / surge_line * 100  # percent, expect negative operating<surge... careful sign

# ---------------------------------------------------------------
# Bootstrap uncertainty on the surge margin curve
# ---------------------------------------------------------------
n_boot = 60
boot_margins = np.zeros((n_boot, len(speed_grid_fine)))
n_pts = len(pr_meas)
for b in range(n_boot):
    idx = rng.integers(0, n_pts, n_pts)
    Xb, yb = X_meas[idx], pr_meas[idx]
    try:
        _, _, PRb = fit_and_reconstruct(Xb, yb, mdot_grid, speed_grid_fine)
        surge_b = np.array([surge_flow_for_speed(mdot_grid, PRb[i, :]) for i in range(len(speed_grid_fine))])
        boot_margins[b] = (operating_line - surge_b) / surge_b * 100
    except Exception:
        boot_margins[b] = np.nan

boot_lo = np.nanpercentile(boot_margins, 5, axis=0)
boot_hi = np.nanpercentile(boot_margins, 95, axis=0)

print("=== Module D4: Compressor Map Reconstruction & Surge Margin ===")
print(f"Measured points: {n_pts} across {len(speed_lines)} speed lines")
print(f"Surge margin at min speed  ({speed_grid_fine[0]:.2f}): {surge_margin[0]:.1f}% "
      f"[{boot_lo[0]:.1f}, {boot_hi[0]:.1f}]% (90% bootstrap band)")
print(f"Surge margin at max speed  ({speed_grid_fine[-1]:.2f}): {surge_margin[-1]:.1f}% "
      f"[{boot_lo[-1]:.1f}, {boot_hi[-1]:.1f}]% (90% bootstrap band)")
min_margin_idx = np.argmin(surge_margin)
print(f"Minimum surge margin: {surge_margin[min_margin_idx]:.1f}% at speed "
      f"{speed_grid_fine[min_margin_idx]:.2f} -> this speed is the operability-critical point")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for s in speed_lines:
    m = np.linspace(0.35, 1.05, 100)
    axes[0, 0].plot(m, true_map(m, s), color="lightgray", lw=1)
axes[0, 0].scatter(mdot_meas, pr_meas, c=speed_meas, cmap="viridis", s=30, edgecolor="k", zorder=5)
axes[0, 0].set_title("Sparse measured points on true speed lines")
axes[0, 0].set_xlabel("corrected mass flow"); axes[0, 0].set_ylabel("pressure ratio")

c1 = axes[0, 1].contourf(MM, SS, PR, levels=20, cmap="viridis")
axes[0, 1].plot(surge_line, speed_grid_fine, color="red", lw=2, label="reconstructed surge line")
axes[0, 1].scatter(mdot_meas, speed_meas, c="white", edgecolor="k", s=20)
axes[0, 1].set_title("Reconstructed full performance map (RBF)")
axes[0, 1].set_xlabel("corrected mass flow"); axes[0, 1].set_ylabel("corrected speed")
axes[0, 1].legend(fontsize=8)
plt.colorbar(c1, ax=axes[0, 1])

axes[1, 0].plot(speed_grid_fine, surge_line, color="red", label="surge line (flow)")
axes[1, 0].plot(speed_grid_fine, operating_line, color="blue", label="assumed operating line")
axes[1, 0].set_title("Surge line vs. operating line")
axes[1, 0].set_xlabel("corrected speed"); axes[1, 0].set_ylabel("corrected mass flow")
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(speed_grid_fine, surge_margin, color="darkgreen", lw=2, label="surge margin")
axes[1, 1].fill_between(speed_grid_fine, boot_lo, boot_hi, color="darkgreen", alpha=0.25, label="90% bootstrap band")
axes[1, 1].axhline(0, color="k", lw=0.8)
axes[1, 1].set_title("Surge margin vs. speed (with reconstruction uncertainty)")
axes[1, 1].set_xlabel("corrected speed"); axes[1, 1].set_ylabel("surge margin [%]")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/moduleD4_compressor_map.png", dpi=150)
print("Saved outputs/moduleD4_compressor_map.png")
