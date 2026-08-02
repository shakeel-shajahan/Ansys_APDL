"""
Case Study 17 : Electro-Chemo-Thermo-Mechanical Lithium-Ion Battery Cell
Beginner demonstration: a single-particle model (SPM) for a spherical active-material
particle. Lithium concentration diffuses radially during a constant-current discharge;
concentration gradients drive both a diffusion-induced stress (chemo-mechanical coupling)
and, through the reaction overpotential, ohmic heat generation (electro-thermal coupling).
"""
import numpy as np

# ---------- particle and material parameters (generic NMC-type particle) ----------
Rp = 5e-6            # particle radius [m]
D = 1e-14             # solid-state diffusivity [m^2/s]
c_max = 48000.0       # maximum lithium concentration [mol/m^3]
c0 = 0.5 * c_max       # initial (50% state of charge)
I_app = 1.0            # applied current density at particle surface [A/m^2] (order 1C for this particle size)
F = 96485.0            # Faraday constant

nr = 31
r = np.linspace(1e-9, Rp, nr)
dr = r[1]-r[0]
dt = 0.2*dr**2/D
n_steps = 4000

c = np.full(nr, c0)
flux_surface = -I_app / F   # negative: lithium is extracted during discharge

for step in range(n_steps):
    c_new = c.copy()
    for i in range(1, nr-1):
        lap = (c[i+1]-2*c[i]+c[i-1])/dr**2 + (2/r[i])*(c[i+1]-c[i-1])/(2*dr)
        c_new[i] = c[i] + D*dt*lap
    c_new[0] = c_new[1]
    # Neumann (flux) boundary condition at the surface
    c_new[-1] = c_new[-2] + flux_surface * dr / D
    c = c_new

soc_local = c / c_max
print("Local state of charge (c/c_max) after discharge, sampled every 5 nodes:")
print(np.round(soc_local[::5], 4))
avg_soc = np.trapezoid(4*np.pi*r**2*c, r) / np.trapezoid(4*np.pi*r**2*np.ones_like(r), r) / c_max
print(f"Volume-averaged state of charge = {avg_soc:.4f}")

# ---------- diffusion-induced stress (Christensen-Newman simplified formula) ----------
Omega_partial = 3.5e-6     # partial molar volume of Li in the host lattice [m^3/mol]
E_particle = 140e9
nu = 0.3
c_avg = np.trapezoid(4*np.pi*r**2*c, r) / np.trapezoid(4*np.pi*r**2*np.ones_like(r), r)
sigma_tangential = (Omega_partial*E_particle)/(3*(1-nu)) * (2*c_avg - c) / c_max * c_max
# (simplified proportional form; full derivation uses the volume-averaged concentration
#  minus the local concentration, scaled by the elastic/molar-volume prefactor)
sigma_tangential = (Omega_partial*E_particle)/(3*(1-nu)*1.0) * (c_avg - c)

print("\nDiffusion-induced tangential stress (MPa), sampled every 5 nodes:")
print(np.round(sigma_tangential[::5]/1e6, 3))
print(f"Maximum diffusion-induced stress magnitude = {np.abs(sigma_tangential).max()/1e6:.2f} MPa")

# ---------- ohmic / reaction heat generation (simple lumped estimate) ----------
R_reaction = 5e-3   # lumped area-specific resistance [Ohm m^2]
q_gen = I_app**2 * R_reaction    # heat generation rate per unit area [W/m^2]
rho_p, cp_p = 2200.0, 700.0
V_over_A = Rp/3    # volume-to-surface-area ratio for a sphere
dTdt = q_gen / (V_over_A * rho_p * cp_p)
print(f"\nHeat generation rate = {q_gen*1e3:.4f} mW/m^2 "
      f"-> lumped particle temperature rise rate = {dTdt*3600:.4f} C/hour")
