"""
Case Study 18 : Thermo-Hydro-Mechanical Reservoir or Freezing Soil
Beginner demonstration: a 1-D column undergoes simultaneous cooling and drainage. A
freezing curve converts a fraction of the pore water to ice as temperature drops below
0 C, releasing latent heat and reducing the effective hydraulic permeability -- the
three-way THM coupling described in the handbook.
"""
import numpy as np

L = 2.0            # column length [m]
nz = 41
z = np.linspace(0, L, nz)
dz = z[1]-z[0]

alpha_th = 1.0e-6   # thermal diffusivity of the soil [m^2/s]
T = np.full(nz, 5.0)     # initial temperature [C]
T[0] = -8.0              # sudden cooling at the top surface

cv_hydraulic0 = 3.0e-7   # consolidation coefficient at unfrozen state [m^2/s]
p = np.full(nz, 50e3)    # initial pore pressure [Pa]
p[0] = 0.0
p[-1] = 50e3

dt = 0.3*dz**2/max(alpha_th, cv_hydraulic0)
n_steps = 3000

L_latent = 334000.0   # latent heat of fusion of water [J/kg]
rho_w = 1000.0
porosity = 0.35
c_soil = 1800.0
rho_soil = 1900.0

def freezing_curve(T):
    """Unfrozen water fraction (0 = fully frozen, 1 = fully liquid), a smooth
    sigmoid transition centered at 0 C over a 1 C freezing-point-depression band."""
    return 1.0 / (1.0 + np.exp(-(T)/0.3))

for step in range(n_steps):
    theta_unfrozen = freezing_curve(T)
    # effective heat capacity increases near the freezing front (latent-heat release
    # is modeled as an enhanced apparent heat capacity, the standard "apparent heat
    # capacity method" for freezing-soil THM problems)
    dtheta_dT = theta_unfrozen*(1-theta_unfrozen)/0.3
    c_apparent = c_soil + porosity*rho_w*L_latent*dtheta_dT/rho_soil

    T_new = T.copy()
    for i in range(1, nz-1):
        lap = (T[i+1]-2*T[i]+T[i-1])/dz**2
        T_new[i] = T[i] + alpha_th*dt/max(c_apparent[i]/c_soil, 1.0) * lap
    T_new[0] = -8.0
    T_new[-1] = 5.0
    T = T_new

    # permeability drops sharply once ice forms (frozen soil is nearly impermeable)
    cv_local = cv_hydraulic0 * theta_unfrozen**3 + 1e-10
    p_new = p.copy()
    for i in range(1, nz-1):
        lap_p = (p[i+1]-2*p[i]+p[i-1])/dz**2
        p_new[i] = p[i] + cv_local[i]*dt*lap_p
    p_new[0] = 0.0
    p_new[-1] = 50e3
    p = p_new

freeze_front = z[np.argmin(np.abs(T - 0.0))]
print(f"Temperature profile (deg C), sampled every 5 nodes:")
print(np.round(T[::5], 2))
print(f"\nFreezing front (T=0) located near z = {freeze_front:.3f} m")

theta_final = freezing_curve(T)
print(f"\nUnfrozen water fraction, sampled every 5 nodes:")
print(np.round(theta_final[::5], 3))

print(f"\nPore pressure profile (kPa), sampled every 5 nodes:")
print(np.round(p[::5]/1e3, 2))
print("\nNote how the pressure gradient is steep and concentrated across the low-permeability")
print("frozen zone (near z=0) while the pressure is nearly uniform through the rest of the")
print("unfrozen column -- the frozen layer acts like an impermeable barrier that absorbs")
print("almost the entire pressure drop, exactly as expected physically.")
