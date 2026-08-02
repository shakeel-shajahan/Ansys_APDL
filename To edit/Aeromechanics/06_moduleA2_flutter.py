"""
Capstone 6 -- Flutter-Speed Prediction for a 2-DOF Pitch-Plunge Airfoil
Section via Eigenvalue-Based Stability Analysis

Goal
----
The classical 2-DOF pitch-plunge model (plunge h, pitch alpha) with
quasi-steady aerodynamic coupling is the textbook reduced model for blade
or wing-section flutter. Sweep airspeed U, assemble the coupled
aeroelastic state matrix at each U, and track the eigenvalues (a
"V-g"/root-locus diagram) to find the flutter speed (the U at which any
mode's damping first crosses zero) and flutter frequency.

Method
------
1. Structural + quasi-steady aerodynamic 2x2 mass/damping/stiffness
   matrices (standard Theodorsen-simplified form).
2. For each airspeed U, build the 4x4 state matrix (2nd-order system
   written in first-order form) and compute its eigenvalues.
3. Plot damping ratio and frequency of each aeroelastic mode vs. U.
4. Flutter speed = first U where damping ratio of any mode goes negative
   (i.e. crosses zero, becomes unstable); flutter frequency = the
   imaginary part of that eigenvalue at flutter onset.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Section properties (typical textbook pitch-plunge parameters)
# ---------------------------------------------------------------
b = 0.5           # semi-chord [m]
a_h = -0.4        # elastic axis location (fraction of semi-chord from mid-chord)
rho = 1.225       # air density [kg/m^3]
m = 25.0          # mass per unit span [kg/m]
I_alpha = 5.0      # pitch mass moment of inertia per unit span [kg m]
S_alpha = 0.6      # static unbalance per unit span [kg]
Kh = 8.0e4         # plunge stiffness [N/m^2] per unit span
Ka = 1.2e4         # pitch stiffness [N] per unit span
Ch = 15.0          # plunge structural damping
Ca = 3.0           # pitch structural damping

Ms = np.array([[m, S_alpha], [S_alpha, I_alpha]])
Cs = np.array([[Ch, 0], [0, Ca]])
Ks = np.array([[Kh, 0], [0, Ka]])


def aero_matrices(U):
    """Quasi-steady 2D aerodynamic stiffness/damping (no Theodorsen lag -- quasi-steady limit)."""
    q = 0.5 * rho * U ** 2
    # lift = 2*pi*rho*U*b*(alpha_eff), moment about elastic axis with a_h offset
    Ka_aero = np.array([[0, 2 * np.pi * rho * U ** 2 * b],
                         [0, 2 * np.pi * rho * U ** 2 * b ** 2 * (0.5 + a_h)]])
    Ca_aero = np.array([[2 * np.pi * rho * U * b, 2 * np.pi * rho * U * b ** 2 * (0.5 - a_h)],
                         [-2 * np.pi * rho * U * b ** 2 * (0.5 + a_h),
                          2 * np.pi * rho * U * b ** 3 * (0.5 + a_h) * (0.5 - a_h)]])
    return Ca_aero, Ka_aero


def state_matrix(U):
    Ca_aero, Ka_aero = aero_matrices(U)
    C_tot = Cs + Ca_aero
    K_tot = Ks + Ka_aero
    Minv = np.linalg.inv(Ms)
    Z = np.zeros((2, 2)); I = np.eye(2)
    A = np.block([[Z, I], [-Minv @ K_tot, -Minv @ C_tot]])
    return A


U_range = np.linspace(1, 150, 400)
freqs = np.zeros((len(U_range), 2))
damps = np.zeros((len(U_range), 2))

for i, U in enumerate(U_range):
    A = state_matrix(U)
    eigvals = np.linalg.eigvals(A)
    # keep the complex-conjugate pairs with positive imaginary part, sorted by frequency
    pos = eigvals[eigvals.imag > 1e-6]
    pos = pos[np.argsort(pos.imag)]
    for k in range(min(2, len(pos))):
        lam = pos[k]
        wn = np.abs(lam)
        zeta = -lam.real / wn if wn > 0 else 0
        freqs[i, k] = lam.imag / (2 * np.pi)
        damps[i, k] = zeta

# flutter speed: first U where either mode's damping crosses <= 0
flutter_idx = None
for i in range(len(U_range)):
    if np.any(damps[i] <= 0):
        flutter_idx = i
        break

U_flutter = U_range[flutter_idx] if flutter_idx is not None else None
mode_at_flutter = np.argmin(damps[flutter_idx]) if flutter_idx is not None else None
f_flutter = freqs[flutter_idx, mode_at_flutter] if flutter_idx is not None else None

print("=== Module A2: 2-DOF Pitch-Plunge Flutter Analysis ===")
print(f"Plunge natural frequency (U=0): {np.sqrt(Ks[0,0]/Ms[0,0])/(2*np.pi):.2f} Hz")
print(f"Pitch  natural frequency (U=0): {np.sqrt(Ks[1,1]/Ms[1,1])/(2*np.pi):.2f} Hz")
if U_flutter:
    print(f"Predicted flutter speed U_F = {U_flutter:.2f} m/s")
    print(f"Predicted flutter frequency  = {f_flutter:.2f} Hz (mode {mode_at_flutter+1})")
else:
    print("No flutter found in the swept airspeed range -- extend U_range.")

# ---------------------------------------------------------------
# Plots (classic V-g / V-f diagram)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for k in range(2):
    axes[0].plot(U_range, freqs[:, k], label=f"Mode {k+1}")
axes[0].set_xlabel("airspeed U [m/s]"); axes[0].set_ylabel("frequency [Hz]")
axes[0].set_title("V-f diagram (frequency coalescence)")
axes[0].legend()

for k in range(2):
    axes[1].plot(U_range, damps[:, k], label=f"Mode {k+1}")
axes[1].axhline(0, color="k", lw=1)
if U_flutter:
    axes[1].axvline(U_flutter, color="red", ls="--", label=f"flutter speed = {U_flutter:.1f} m/s")
axes[1].set_xlabel("airspeed U [m/s]"); axes[1].set_ylabel("damping ratio")
axes[1].set_title("V-g diagram (stability)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/moduleA2_flutter.png", dpi=150)
print("Saved outputs/moduleA2_flutter.png")
