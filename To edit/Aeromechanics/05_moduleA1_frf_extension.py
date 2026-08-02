"""
Extension to Capstone 5 -- Modal Assurance Criterion (MAC) and a
Stabilization Diagram for Confident Mode Identification

Addresses reviewer feedback: "multiple sensors, MAC, stabilization
diagram, PolyMAX, SSI-COV" beyond the single-sensor H1/peak-picking
capstone.

Part A: extend the single-sensor (DOF 1) FRF test to all 3 DOFs, and
compute the Modal Assurance Criterion (MAC) between the identified mode
shapes (from a simple residue-based shape extraction at each resonance)
and the TRUE mode shapes, to quantify shape-identification confidence --
not just frequency accuracy.

Part B: a genuine stabilization diagram -- fit a simple parametric
(rational-fraction-polynomial-style, via Prony/least-squares complex
exponential fitting) modal model at INCREASING model order, and track
which identified poles are numerically "stable" (repeat within tight
tolerance) across orders vs. which are spurious computational modes --
the standard experimental-modal-analysis engineering judgment tool that
peak-picking alone cannot provide.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.signal import csd, welch

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "solve.py")).read().split(
    "# ---------------------------------------------------------------\n# Plots"
)[0])

# ---------------------------------------------------------------
# Part A: 3-sensor FRFs + MAC between identified and true mode shapes
# ---------------------------------------------------------------
resp_channels = sol.y[:3]  # displacement at all 3 DOFs (full 3-DOF response already simulated)
resp_meas_all = resp_channels + rng.normal(0, meas_noise * resp_channels.std(), resp_channels.shape)

H1_all = []
for dof in range(3):
    _, Sfx_d = csd(force_meas, resp_meas_all[dof], fs=fs, nperseg=nperseg)
    H1_all.append(Sfx_d / Sff)
H1_all = np.array(H1_all)  # shape (3, n_freq)

# identified shape at each resonance = residue vector (H1 magnitude across the 3 sensors, at that freq bin)
identified_shapes = []
for fn, _ in identified[:3]:
    idx = np.argmin(np.abs(f_csd - fn))
    shape = np.abs(H1_all[:, idx])
    shape = shape / np.max(shape)
    identified_shapes.append(shape)
identified_shapes = np.array(identified_shapes)  # (n_modes, 3)

# true mode shapes (from the eigen-decomposition already done in solve.py: Phi, order)
Phi_true_modes = Phi[:, :len(identified_shapes)]
Phi_true_modes = Phi_true_modes / np.max(np.abs(Phi_true_modes), axis=0)


def mac(phi_a, phi_b):
    num = np.abs(phi_a @ phi_b) ** 2
    den = (phi_a @ phi_a) * (phi_b @ phi_b)
    return num / den


mac_matrix = np.zeros((len(identified_shapes), len(identified_shapes)))
for i in range(len(identified_shapes)):
    for j in range(len(identified_shapes)):
        mac_matrix[i, j] = mac(identified_shapes[i], np.abs(Phi_true_modes[:, j]))

print("=== Extension 5a: Modal Assurance Criterion (MAC) ===")
print("MAC matrix (rows: identified modes, cols: true modes) -- diagonal should be close to 1:")
print(np.round(mac_matrix, 3))
diag_mac = np.diag(mac_matrix)
print(f"Diagonal MAC values (shape-identification confidence): {np.round(diag_mac, 3)}")

# ---------------------------------------------------------------
# Part B: Stabilization diagram via least-squares complex-exponential
# (Prony-style) fitting of the impulse response at increasing model order
# ---------------------------------------------------------------
from statsmodels.regression.linear_model import yule_walker
from scipy.signal import decimate

decim_factor = 4
fs_eff = fs / decim_factor  # 500 Hz effective rate
x1_d = decimate(x1_meas, decim_factor, ftype="fir")
x1_d = (x1_d - x1_d.mean()) / np.std(x1_d)

orders = list(range(4, 31, 2))
stab_freqs = {}
for order in orders:
    rho, _ = yule_walker(x1_d, order=order, method="mle")
    char_poly = np.concatenate([[1.0], -rho])
    roots = np.roots(char_poly)
    mask = (np.abs(roots) < 1.0) & (roots.imag > 1e-6)
    r = roots[mask]
    freqs_here = np.angle(r) * fs_eff / (2 * np.pi)
    freqs_here = freqs_here[(freqs_here > 2) & (freqs_here < 25)]
    stab_freqs[order] = sorted(freqs_here)

print("\n=== Extension 5b: Stabilization Diagram (AR-model pole tracking) ===")
for order in sorted(stab_freqs):
    print(f"Order {order:2d}: identified frequencies = {np.round(stab_freqs[order], 1)}")

# stable poles = frequencies that repeat within 1% across at least 3 consecutive orders
def find_stable_poles(stab_freqs, tol=0.06, min_repeats=5):
    orders = sorted(stab_freqs)
    all_candidates = sorted(set(f for fl in stab_freqs.values() for f in fl))
    stable = []
    for f in all_candidates:
        count = sum(1 for o in orders if any(abs(ff - f) / f < tol for ff in stab_freqs[o]))
        if count >= min_repeats:
            stable.append(f)
    # cluster nearby candidates
    stable = sorted(stable)
    clustered = []
    for f in stable:
        if not clustered or abs(f - clustered[-1]) / f > tol * 2:
            clustered.append(f)
    return clustered


stable_poles = find_stable_poles(stab_freqs)
print(f"\nPhysically stable poles (repeat across >=3 orders): {np.round(stable_poles, 2)} Hz")
print(f"(True modal frequencies for reference: {np.round(wn_true, 2)} Hz)")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

im = axes[0].imshow(mac_matrix, cmap="viridis", vmin=0, vmax=1)
axes[0].set_xticks(range(len(identified_shapes))); axes[0].set_yticks(range(len(identified_shapes)))
axes[0].set_xticklabels([f"true {i+1}" for i in range(len(identified_shapes))])
axes[0].set_yticklabels([f"ident. {i+1}" for i in range(len(identified_shapes))])
for i in range(len(identified_shapes)):
    for j in range(len(identified_shapes)):
        axes[0].text(j, i, f"{mac_matrix[i,j]:.2f}", ha="center", va="center",
                     color="white" if mac_matrix[i, j] < 0.5 else "black", fontsize=9)
axes[0].set_title("MAC matrix: identified vs. true mode shapes")
plt.colorbar(im, ax=axes[0])

for order in sorted(stab_freqs):
    freqs_here = stab_freqs[order]
    axes[1].scatter(freqs_here, [order] * len(freqs_here), color="steelblue", s=15)
for fw in stable_poles:
    axes[1].axvline(fw, color="red", ls="--", lw=0.7, alpha=0.7)
for fw in wn_true:
    axes[1].axvline(fw, color="green", ls=":", lw=1.2, alpha=0.7)
axes[1].set_title("Stabilization diagram (blue=poles per order, red=stable, green=true)")
axes[1].set_xlabel("frequency [Hz]"); axes[1].set_ylabel("AR model order")
axes[1].set_xlim(0, 25)

plt.tight_layout()
plt.savefig(os.path.join(here, "outputs", "moduleA1_extension_mac_stabilization.png"), dpi=150)
print("Saved outputs/moduleA1_extension_mac_stabilization.png")
