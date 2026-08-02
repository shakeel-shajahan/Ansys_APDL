"""
Case Study 6 : Hydrogel / Microgel Swelling in Solvent
Beginner demonstration: a spherically symmetric gel particle absorbs solvent by Fickian
diffusion; local concentration increase drives a volumetric swelling strain and, through
a simple neo-Hookean-like law, a hoop stress. This shows the two-way link: concentration
changes stress state and (in the full model) stress/deformation changes the diffusivity.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- radial Fickian diffusion in a sphere (finite difference) ----------
R = 1.0e-3          # initial particle radius [m]
D = 2.0e-11         # solvent diffusivity in the gel [m^2/s]
c_env = 1.0         # normalized environmental (bath) concentration
c0 = 0.0            # initial concentration inside the gel

nr = 41
r = np.linspace(1e-9, R, nr)     # avoid r=0 singularity in 1/r term
dr = r[1] - r[0]
dt = 0.2 * dr**2 / D
n_steps = 3000

c = np.full(nr, c0)
c[-1] = c_env    # Dirichlet: bath concentration imposed at the gel surface

for step in range(n_steps):
    c_new = c.copy()
    for i in range(1, nr-1):
        lap = (c[i+1] - 2*c[i] + c[i-1]) / dr**2 + (2/r[i]) * (c[i+1]-c[i-1]) / (2*dr)
        c_new[i] = c[i] + D * dt * lap
    c_new[0] = c_new[1]     # symmetry at the center
    c_new[-1] = c_env
    c = c_new

print("Concentration profile at final time (normalized), sampled every 10 nodes:")
print(np.round(c[::10], 4))

# ---------- volumetric swelling strain and hoop stress (Flory-Rehner-like, simplified) ----------
Omega = 0.4        # normalized swelling coefficient (dimensionless concentration scale 0-1)
J = 1 + Omega * c  # simple linear swelling law: J = 1 + Omega*c  (volumetric stretch)
G = 5.0e4          # shear modulus of the dry network [Pa]
# neo-Hookean-like radial/hoop stress difference driven by local swelling mismatch
sigma_hoop = G * (J**(2/3) - J**(-4/3))

print("\nVolumetric swelling ratio J at final time (sampled every 10 nodes):")
print(np.round(J[::10], 4))
print("\nHoop stress (kPa) at final time (sampled every 10 nodes):")
print(np.round(sigma_hoop[::10]/1e3, 3))

# ---------- mass conservation check ----------
total_solvent = np.trapezoid(4*np.pi*r**2*c, r)
print(f"\nTotal absorbed solvent proxy (should grow monotonically as diffusion proceeds): "
      f"{total_solvent:.4e}")

t_char = R**2 / D
print(f"Characteristic diffusion time R^2/D = {t_char:.1f} s "
      f"= {t_char/3600:.2f} h  (rule-of-thumb equilibration time)")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(r*1e3, c); ax[0].set_xlabel("r [mm]"); ax[0].set_ylabel("concentration (norm.)")
ax[1].plot(r*1e3, sigma_hoop/1e3); ax[1].set_xlabel("r [mm]"); ax[1].set_ylabel("hoop stress [kPa]")
plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch06.png", dpi=150)
print("Figure saved.")
