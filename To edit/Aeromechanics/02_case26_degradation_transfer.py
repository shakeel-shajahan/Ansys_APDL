"""
Capstone 2 -- Physically-Parameterised Degradation and Cross-Fidelity
Calibration (inspired by NASA Rotor 67 style tip-clearance/erosion studies)

Goal
----
An LP steam-turbine last stage loses efficiency in service through two
physical mechanisms: (a) tip-clearance growth (rub wear, thermal
transients) and (b) blade-surface erosion/roughening from wet-steam
droplet impact. We build a compact *physical* degradation model for
efficiency loss as a function of these two parameters, generate a
"high-fidelity" (expensive 3D RANS-like) dataset and a cheaper
"low-fidelity" (e.g. mean-line / correlation-based) dataset that shares
the same functional shape but is biased, and calibrate the low-fidelity
model to the high-fidelity data using a small number of high-fidelity
anchor points -- the standard multi-fidelity workflow used when full
CFD campaigns are too expensive to run for every degradation state.

Method
------
1. Ground truth: efficiency loss = f(tip_clearance, erosion) from a
   physically motivated model (linear + interaction + mild saturation).
2. High-fidelity ("HF") data: ground truth + small noise, expensive,
   n_hf = 8 points only.
3. Low-fidelity ("LF") data: ground truth passed through a *biased*
   surrogate correlation (systematic multiplicative + additive bias),
   cheap, n_lf = 200 points (dense sweep).
4. Calibrate: fit a simple affine correction  y_hf ~= a * y_lf + b  using
   the 8 HF anchor points (paired with LF predictions at the same
   inputs), then apply the correction to the full dense LF sweep.
5. Report calibration gap before/after, and map full degradation surface.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(26)

# ---------------------------------------------------------------
# 1. Ground-truth physical degradation model
# ---------------------------------------------------------------
def true_efficiency_loss(tip_clear_mm, erosion_level):
    """tip_clear_mm in [0, 2.0] mm growth, erosion_level in [0, 1] (0=new, 1=fully roughened)."""
    loss = (1.6 * tip_clear_mm
            + 4.0 * erosion_level
            + 1.1 * tip_clear_mm * erosion_level
            - 0.35 * tip_clear_mm ** 2)
    return np.clip(loss, 0, None)  # percentage points of efficiency lost

def lf_correlation(tip_clear_mm, erosion_level):
    """Cheap mean-line correlation: same functional family but biased (under-predicts
    interaction effect and has a constant offset -- a realistic mean-line shortfall)."""
    loss = (1.35 * tip_clear_mm + 3.3 * erosion_level + 0.25 * tip_clear_mm * erosion_level)
    return loss + 0.4  # systematic low-order-model offset

# ---------------------------------------------------------------
# 2. Data generation
# ---------------------------------------------------------------
n_hf = 8
tc_hf = rng.uniform(0.1, 1.8, n_hf)
er_hf = rng.uniform(0.05, 0.95, n_hf)
y_hf_true = true_efficiency_loss(tc_hf, er_hf) + rng.normal(0, 0.08, n_hf)

n_lf = 400
tc_lf = rng.uniform(0, 2.0, n_lf)
er_lf = rng.uniform(0, 1.0, n_lf)
y_lf = lf_correlation(tc_lf, er_lf) + rng.normal(0, 0.05, n_lf)

y_lf_at_hf = lf_correlation(tc_hf, er_hf)

# ---------------------------------------------------------------
# 3. Affine calibration a*y_lf + b  fit on the 8 HF anchor points
# ---------------------------------------------------------------
A = np.column_stack([y_lf_at_hf, np.ones(n_hf)])
coef, *_ = np.linalg.lstsq(A, y_hf_true, rcond=None)
a_cal, b_cal = coef
y_lf_calibrated = a_cal * y_lf + b_cal

# ---------------------------------------------------------------
# 4. Evaluate calibration quality on a dense grid vs. true ground truth
# ---------------------------------------------------------------
tc_grid = np.linspace(0, 2.0, 50)
er_grid = np.linspace(0, 1.0, 50)
TC, ER = np.meshgrid(tc_grid, er_grid)
Y_true = true_efficiency_loss(TC, ER)
Y_lf_raw = lf_correlation(TC, ER)
Y_lf_cal = a_cal * Y_lf_raw + b_cal

rmse_raw = np.sqrt(np.mean((Y_lf_raw - Y_true) ** 2))
rmse_cal = np.sqrt(np.mean((Y_lf_cal - Y_true) ** 2))

print("=== Case 26: Degradation & Cross-Fidelity Calibration ===")
print(f"HF anchor points used for calibration : {n_hf}")
print(f"Calibration fit: y_hf ~= {a_cal:.3f} * y_lf + {b_cal:.3f}")
print(f"Uncalibrated LF-vs-truth RMSE (eff. pts): {rmse_raw:.3f}")
print(f"Calibrated   LF-vs-truth RMSE (eff. pts): {rmse_cal:.3f}")
print(f"Calibration reduced RMSE by {100*(1-rmse_cal/rmse_raw):.1f}%")

# ---------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
levels = np.linspace(0, max(Y_true.max(), Y_lf_raw.max()), 20)

c0 = axes[0, 0].contourf(TC, ER, Y_true, levels=levels, cmap="inferno")
axes[0, 0].scatter(tc_hf, er_hf, c="cyan", edgecolor="k", s=50, label="8 HF (CFD) anchors")
axes[0, 0].set_title("True efficiency loss (HF ground truth)")
axes[0, 0].set_xlabel("tip clearance growth [mm]"); axes[0, 0].set_ylabel("erosion level [-]")
axes[0, 0].legend(fontsize=8)
plt.colorbar(c0, ax=axes[0, 0])

c1 = axes[0, 1].contourf(TC, ER, Y_lf_raw, levels=levels, cmap="inferno")
axes[0, 1].set_title("Raw low-fidelity correlation (biased)")
axes[0, 1].set_xlabel("tip clearance growth [mm]"); axes[0, 1].set_ylabel("erosion level [-]")
plt.colorbar(c1, ax=axes[0, 1])

c2 = axes[1, 0].contourf(TC, ER, Y_lf_cal, levels=levels, cmap="inferno")
axes[1, 0].scatter(tc_hf, er_hf, c="cyan", edgecolor="k", s=50)
axes[1, 0].set_title("Calibrated low-fidelity map")
axes[1, 0].set_xlabel("tip clearance growth [mm]"); axes[1, 0].set_ylabel("erosion level [-]")
plt.colorbar(c2, ax=axes[1, 0])

axes[1, 1].bar(["Raw LF\nRMSE", "Calibrated LF\nRMSE"], [rmse_raw, rmse_cal], color=["firebrick", "seagreen"])
axes[1, 1].set_ylabel("efficiency-loss RMSE [pts]")
axes[1, 1].set_title("Effect of 8-point affine calibration")
for i, v in enumerate([rmse_raw, rmse_cal]):
    axes[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("outputs/case26_degradation_calibration.png", dpi=150)
print("Saved outputs/case26_degradation_calibration.png")
