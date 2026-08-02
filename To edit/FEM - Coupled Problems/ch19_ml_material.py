"""
Case Study 19 : Physics-Constrained Learned Material Model inside FSI
Beginner demonstration: we fit a small neural network correction on top of a neo-Hookean
base model to reproduce synthetic "experimental" uniaxial stress-stretch data, using only
invariants (I1) as input so the model is automatically frame-invariant. We then check the
resulting tangent modulus stays positive (a basic stability/consistency requirement before
such a model could ever be used inside an FE/FSI Newton solve).
"""
import numpy as np

np.random.seed(0)

# ---------- synthetic "experimental" data from a Gent hyperelastic model ----------
def gent_stress(lam, mu=1.2e6, Jm=60.0):
    """Uniaxial nominal (engineering) stress for a Gent model."""
    I1 = lam**2 + 2/lam
    factor = Jm / (Jm - (I1 - 3))
    return mu * factor * (lam - 1/lam**2)

lam_data = np.linspace(1.01, 2.5, 40)
stress_data = gent_stress(lam_data) * (1 + 0.01*np.random.randn(len(lam_data)))  # add noise

# ---------- base (neo-Hookean) model the student already trusts ----------
mu_base = 1.0e6
def neo_hookean_stress(lam):
    return mu_base * (lam - 1/lam**2)

residual = stress_data - neo_hookean_stress(lam_data)

# ---------- tiny 1-hidden-layer neural network correction, trained on I1 only ----------
I1_data = (lam_data**2 + 2/lam_data - 3).reshape(-1, 1)   # invariant, shifted so I1=0 at rest
y = residual.reshape(-1, 1) / 1e6   # scale to O(1) for stable training

rng = np.random.default_rng(1)
n_hidden = 6
W1 = rng.normal(scale=0.5, size=(1, n_hidden))
b1 = np.zeros((1, n_hidden))
W2 = rng.normal(scale=0.5, size=(n_hidden, 1))
b2 = np.zeros((1, 1))

def forward(I1):
    z1 = I1 @ W1 + b1
    a1 = np.tanh(z1)
    out = a1 @ W2 + b2
    return out, a1, z1

lr = 0.05
for epoch in range(4000):
    out, a1, z1 = forward(I1_data)
    err = out - y
    loss = (err**2).mean()
    dW2 = a1.T @ err / len(y)
    db2 = err.mean(axis=0, keepdims=True)
    da1 = err @ W2.T
    dz1 = da1 * (1 - a1**2)
    dW1 = I1_data.T @ dz1 / len(y)
    db1 = dz1.mean(axis=0, keepdims=True)
    W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2

print(f"Final training MSE (scaled units) = {loss:.6f}")

def learned_stress(lam):
    I1 = (lam**2 + 2/lam - 3)
    corr, _, _ = forward(np.array([[I1]]))
    return neo_hookean_stress(lam) + corr[0,0]*1e6

print("\nComparison at a few stretch levels (MPa): true Gent  vs  base neo-Hookean  vs  ML-corrected")
for lam_test in [1.2, 1.6, 2.0, 2.4]:
    s_true = gent_stress(lam_test)/1e6
    s_base = neo_hookean_stress(lam_test)/1e6
    s_ml = learned_stress(lam_test)/1e6
    print(f"  lambda={lam_test:.1f}:  Gent={s_true:6.3f}  neo-Hookean={s_base:6.3f}  "
          f"ML-corrected={s_ml:6.3f}")

# ---------- consistency check: tangent modulus must stay positive (dsigma/dlambda > 0) ----------
lam_check = np.linspace(1.01, 2.6, 60)
d_lam = 1e-4
tangents = [(learned_stress(l+d_lam) - learned_stress(l-d_lam))/(2*d_lam) for l in lam_check]
tangents = np.array(tangents)
print(f"\nMinimum tangent modulus over the tested range = {tangents.min()/1e6:.4f} MPa "
      f"({'OK: stays positive, Newton-friendly' if tangents.min() > 0 else 'WARNING: non-physical softening detected'})")

print("\nExtrapolation warning check (beyond the training range, lambda=3.0):")
s_extrap = learned_stress(3.0)/1e6
print(f"  ML-corrected stress at lambda=3.0 (never seen in training) = {s_extrap:.3f} MPa "
      f"-- always validate a learned model outside its training envelope before using it in FSI.")
