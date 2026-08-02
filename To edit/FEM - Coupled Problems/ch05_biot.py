"""
Case Study 5 : Biot Consolidation with Heterogeneous Permeability
Beginner demonstration: 1-D Terzaghi consolidation, solved (a) with the classical
analytical Fourier series and (b) with an explicit finite-difference scheme for the
Biot pressure equation, then compared. This is the validation ladder in action.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Terzaghi analytical solution ----------
def terzaghi_pressure(z, t, H, cv, p0, n_terms=60):
    """z in [0,H], drained at both ends (p=0), uniform initial excess pressure p0.
    Fourier sine series with only ODD harmonics n=1,3,5,... contribute because the
    initial condition is a constant p0 (even harmonics integrate to zero)."""
    p = np.zeros_like(z)
    for m in range(n_terms):
        n = 2*m + 1
        p += (4*p0/(n*np.pi)) * np.sin(n*np.pi*z/H) * np.exp(-(n*np.pi/H)**2 * cv * t)
    return p

H = 5.0            # drainage path length [m]
cv = 2.0e-7        # coefficient of consolidation [m^2/s]
p0 = 100e3         # initial excess pore pressure [Pa]
z = np.linspace(0, H, 51)
times = [1e4, 1e5, 1e6, 1e7]

print("Terzaghi analytical excess pore pressure at mid-depth (z=H/2), kPa:")
for tt in times:
    pm = terzaghi_pressure(np.array([H/2]), tt, H, cv, p0)[0]
    Tv = cv*tt/H**2
    print(f"  t = {tt:.0e} s  (Tv={Tv:.3f})  ->  p = {pm/1e3:.2f} kPa")

# ---------- explicit finite-difference Biot pressure diffusion (fixed-stress split) ----------
nz = 51
dz = H/(nz-1)
dt_fd = 0.4*dz**2/cv
n_steps = 4000
p_fd = np.full(nz, p0)
p_fd[0] = 0.0
p_fd[-1] = 0.0

for step in range(n_steps):
    p_new = p_fd.copy()
    for i in range(1, nz-1):
        p_new[i] = p_fd[i] + cv*dt_fd/dz**2 * (p_fd[i+1] - 2*p_fd[i] + p_fd[i-1])
    p_new[0] = 0.0
    p_new[-1] = 0.0
    p_fd = p_new

t_final = n_steps*dt_fd
p_analytical_final = terzaghi_pressure(z, t_final, H, cv, p0)
rel_err = np.abs(p_fd - p_analytical_final).max() / p0
print(f"\nFinite-difference vs analytical at t = {t_final:.3e} s: "
      f"max relative error = {rel_err*100:.3f} %")

# ---------- settlement (mass balance check) ----------
settlement_fd = np.trapezoid(p0 - p_fd, z) / p0 * H   # normalized cumulative volume change proxy
settlement_an = np.trapezoid(p0 - p_analytical_final, z) / p0 * H
print(f"Normalized settlement proxy: FD = {settlement_fd:.4f} m, analytical = {settlement_an:.4f} m")

fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(z, p_fd/1e3, 'o-', ms=3, label="finite difference")
ax.plot(z, p_analytical_final/1e3, '--', label="Terzaghi analytical")
ax.set_xlabel("depth z [m]"); ax.set_ylabel("excess pore pressure [kPa]")
ax.legend(); plt.tight_layout()
plt.savefig("/home/claude/coupled_book/figures/ch05.png", dpi=150)
print("Figure saved.")
