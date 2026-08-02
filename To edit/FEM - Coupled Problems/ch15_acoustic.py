"""
Case Study 15 : Acoustic-Structure Interaction of a Submerged Plate
Beginner demonstration: a single-mode plate oscillator coupled to a lumped acoustic
"radiation mass" (added mass from the surrounding fluid) reproduces the classic wet
vs dry natural-frequency shift, plus a simple radiated-power estimate.
"""
import numpy as np

# ---------- plate modal parameters (dry, in vacuum) ----------
m_plate = 2.5        # modal mass [kg]
k_plate = 3.5e5      # modal stiffness [N/m]
wn_dry = np.sqrt(k_plate / m_plate)
print(f"Dry natural frequency (in vacuum) = {wn_dry/2/np.pi:.2f} Hz")

# ---------- added mass from the acoustic fluid (classical added-mass formula for a
# baffled circular piston radiating into a heavy fluid, water) ----------
rho_water = 1000.0
a_plate = 0.15        # equivalent radius of the vibrating plate [m]
m_added = 8.0/3.0 * rho_water * a_plate**3   # Rayleigh's added mass of a baffled piston

wn_wet = np.sqrt(k_plate / (m_plate + m_added))
print(f"Acoustic added mass = {m_added:.3f} kg "
      f"({m_added/m_plate*100:.1f}% of the structural mass)")
print(f"Wet natural frequency (submerged in water) = {wn_wet/2/np.pi:.2f} Hz")
print(f"Frequency shift due to fluid loading = "
      f"{(wn_dry-wn_wet)/wn_dry*100:.1f}% reduction")

# ---------- radiation damping and radiated acoustic power (piston-in-baffle model) ----------
c_water = 1480.0     # speed of sound in water [m/s]
k_wave = wn_wet / c_water
ka = k_wave * a_plate
# radiation resistance coefficient (low ka limit, Rayleigh piston)
R_rad = rho_water * c_water * np.pi * a_plate**2 * (ka**2 / 2) if ka < 1 else \
        rho_water * c_water * np.pi * a_plate**2

v_amp = 0.02   # plate velocity amplitude [m/s]
P_radiated = 0.5 * R_rad * v_amp**2
print(f"\nRadiation regime parameter ka = {ka:.4f} "
      f"({'compact source, low ka' if ka<1 else 'radiating efficiently, ka>1'})")
print(f"Estimated radiated acoustic power = {P_radiated*1e3:.4f} mW "
      f"for a plate velocity amplitude of {v_amp} m/s")

print("\nAir-loading comparison (added mass is negligible for a plate vibrating in air):")
rho_air = 1.225
m_added_air = 8.0/3.0 * rho_air * a_plate**3
wn_air = np.sqrt(k_plate / (m_plate + m_added_air))
print(f"  added mass in air = {m_added_air:.6f} kg -> "
      f"wet(air) frequency = {wn_air/2/np.pi:.2f} Hz "
      f"(negligible shift of {(wn_dry-wn_air)/wn_dry*100:.4f}%)")
