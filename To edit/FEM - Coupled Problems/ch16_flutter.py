"""
Case Study 16 : Aeroelastic Flutter of the AGARD 445.6 Wing
Beginner demonstration: the classical 2-DOF "typical section" model (plunge h, pitch
alpha) with quasi-steady aerodynamics reproduces flutter onset as airspeed increases --
the same coupled-eigenvalue idea used for the full AGARD wing, but at desk-calculator
scale.
"""
import numpy as np

# ---------- structural section properties (illustrative, generic thin airfoil) ----------
m = 20.0            # mass per unit span [kg/m]
S = 2.0              # static mass moment [kg]
I_alpha = 3.0        # mass moment of inertia [kg m]
k_h = 3.0e4          # plunge stiffness [N/m^2]
k_alpha = 2.5e3      # pitch stiffness [N]
b = 0.5              # half-chord [m]
rho = 1.225          # air density [kg/m^3]
a = -0.2             # elastic axis location (dimensionless, mid-chord = 0)

def sweep(U_range):
    for U in U_range:
        M = np.array([[m, S], [S, I_alpha]])
        L_alpha = 2*np.pi*rho*U**2*b
        M_alpha = 2*np.pi*rho*U**2*b**2*(0.5+a)
        C_aero = np.array([[0, -2*np.pi*rho*U*b], [0, 2*np.pi*rho*U*b**2*(0.5+a)]])
        K = np.array([[k_h, L_alpha], [0, k_alpha - M_alpha]])
        Minv = np.linalg.inv(M)
        Z = np.zeros((2, 2)); I2 = np.eye(2)
        A_state = np.block([[Z, I2], [-Minv @ K, -Minv @ C_aero]])
        eigvals = np.linalg.eigvals(A_state)
        yield U, eigvals

print("Airspeed sweep: maximum real part of the aeroelastic eigenvalues")
print("(positive real part = growing oscillation = flutter):\n")
U_flutter = None
for U, eigvals in sweep(np.linspace(5, 150, 146)):
    max_real = eigvals.real.max()
    tag = ""
    if max_real > 0 and U_flutter is None:
        U_flutter = U
        tag = "  <-- FLUTTER ONSET"
    if int(round(U)) % 20 == 0 or tag:
        pos_imag = eigvals[eigvals.imag > 0]
        freq_hz = abs(pos_imag[0].imag)/(2*np.pi) if len(pos_imag) else float("nan")
        print(f"  U = {U:5.1f} m/s   max Re(eigval) = {max_real:8.4f}   "
              f"osc. freq ~ {freq_hz:.2f} Hz{tag}")

print(f"\nEstimated flutter speed (onset of dynamic instability) = {U_flutter:.1f} m/s")
wn_h = np.sqrt(k_h/m); wn_alpha = np.sqrt(k_alpha/I_alpha)
print(f"Uncoupled plunge frequency = {wn_h/2/np.pi:.2f} Hz, "
      f"uncoupled pitch frequency = {wn_alpha/2/np.pi:.2f} Hz")
print("Flutter occurs when aerodynamic coupling merges these two modes into a single")
print("unstable branch, exactly the mechanism the handbook describes for AGARD 445.6.")
