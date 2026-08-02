"""
Case Study 4 : Magneto-Electro-Mechanical Composite RVE
Beginner demonstration: two-phase laminate RVE (piezoelectric + magnetostrictive layers).
We compute effective coupled properties with the classical series/parallel (Voigt/Reuss)
mixing rules and verify the Hill-Mandel energy consistency condition on a simple example.
"""
import numpy as np

# ---------- phase properties (illustrative) ----------
# Phase A: piezoelectric ceramic, Phase B: magnetostrictive alloy
vf_A = 0.5          # volume fraction of phase A
vf_B = 1 - vf_A

C_A, C_B = 60e9, 45e9        # elastic stiffness [Pa]
e_A, e_B = 12.0, 0.0          # piezoelectric coupling [C/m^2]  (B has none)
q_A, q_B = 0.0, 18.0          # piezomagnetic coupling [N/(A m)] (A has none)
kappa_A, kappa_B = 1.5e-8, 8e-9   # dielectric permittivity [F/m]
mu_A, mu_B = 1.1e-6, 4.5e-5       # magnetic permeability [H/m]

def voigt(pA, pB):
    """Iso-strain (parallel / series-layer, loading along the layers)."""
    return vf_A * pA + vf_B * pB

def reuss(pA, pB):
    """Iso-stress (loading across the layers)."""
    return 1.0 / (vf_A / pA + vf_B / pB)

C_eff_voigt = voigt(C_A, C_B)
C_eff_reuss = reuss(C_A, C_B)
e_eff = voigt(e_A, e_B)     # coupling coefficients mix like Voigt for in-plane loading
q_eff = voigt(q_A, q_B)
kappa_eff = voigt(kappa_A, kappa_B)
mu_eff = voigt(mu_A, mu_B)

print("Effective (Voigt, iso-strain) stiffness  :", round(C_eff_voigt/1e9, 2), "GPa")
print("Effective (Reuss, iso-stress) stiffness  :", round(C_eff_reuss/1e9, 2), "GPa")
print("Effective piezoelectric coupling e_eff    :", round(e_eff, 3), "C/m^2")
print("Effective piezomagnetic coupling q_eff    :", round(q_eff, 3), "N/(A m)")
print("Note: a nonzero magnetoelectric coefficient alpha_ME can only emerge from the")
print("      PRODUCT of e and q terms in a fully coupled homogenization; the simple")
print("      linear mixing rules above give zero directly, motivating the need for a")
print("      full coupled RVE solve (FEniCSx / deal.II) rather than naive averaging.")

# ---------- Hill-Mandel energy consistency check on a toy periodic strain field ----------
np.random.seed(0)
n = 200
# generate a synthetic periodic microstrain field with zero mean (illustrative only)
strain_field = 0.01 * (np.random.rand(n) - 0.5)
strain_field -= strain_field.mean()
stress_field = np.where(np.arange(n) < n * vf_A, C_A, C_B) * strain_field

macro_strain = strain_field.mean()
macro_stress = stress_field.mean()
micro_energy = (stress_field * strain_field).mean()
macro_energy = macro_stress * macro_strain

print(f"\nHill-Mandel check: <sigma:eps> = {micro_energy:.6e}, "
      f"<sigma>:<eps> = {macro_energy:.6e}")
print("These differ because this random field does not satisfy periodic admissibility;")
print("a genuine RVE solve enforces periodic boundary conditions so the two match exactly.")
