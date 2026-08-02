"""
Case Study 20 : Neural Operator Benchmark for Coupled PDEs
Beginner demonstration: instead of a full FNO/DeepONet, we train a small feed-forward
network to map a scalar parameter (a heat-source position) directly to the STEADY-STATE
temperature field of Case Study 1's 1D bar -- the simplest possible "operator learning"
task: parameter -> field. We compare against the true finite-difference solution and
check generalization to an unseen parameter value.
"""
import numpy as np

# ---------- generate training data: solve the steady 1D heat equation for many source positions ----------
nx = 41
x = np.linspace(0, 1, nx)
dx = x[1]-x[0]

def true_field(x_source, amplitude=50.0, width=0.05):
    """Exact (up to discretization) steady solution of -d2T/dx2 = f(x), T(0)=T(1)=0,
    solved directly with a tridiagonal finite-difference system. f is a Gaussian source
    centered at x_source. This is the trusted numerical 'ground truth' for training data."""
    f = amplitude * np.exp(-(x - x_source)**2 / (2*width**2))
    n_int = nx - 2
    A = (np.diag(2*np.ones(n_int)) + np.diag(-1*np.ones(n_int-1), 1)
         + np.diag(-1*np.ones(n_int-1), -1)) / dx**2
    rhs = f[1:-1]
    T_int = np.linalg.solve(A, rhs)
    T = np.zeros_like(x)
    T[1:-1] = T_int
    return T

train_positions = np.linspace(0.15, 0.85, 15)
X_train = train_positions.reshape(-1, 1)
Y_train = np.array([true_field(xs) for xs in train_positions])   # shape (15, 41)

# normalize
Y_mean, Y_std = Y_train.mean(), Y_train.std()
Y_train_n = (Y_train - Y_mean) / Y_std

# ---------- tiny operator network: 1 input -> hidden -> 41 outputs (the whole field at once) ----------
rng = np.random.default_rng(0)
n_hidden = 20
W1 = rng.normal(scale=0.5, size=(1, n_hidden))
b1 = np.zeros(n_hidden)
W2 = rng.normal(scale=0.3, size=(n_hidden, nx))
b2 = np.zeros(nx)

def forward(Xp):
    z1 = Xp @ W1 + b1
    a1 = np.tanh(z1)
    out = a1 @ W2 + b2
    return out, a1

lr = 0.02
for epoch in range(6000):
    out, a1 = forward(X_train)
    err = out - Y_train_n
    loss = (err**2).mean()
    dW2 = a1.T @ err / len(X_train)
    db2 = err.mean(axis=0)
    da1 = err @ W2.T
    dz1 = da1 * (1 - a1**2)
    dW1 = X_train.T @ dz1 / len(X_train)
    db1 = dz1.mean(axis=0)
    W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2

print(f"Final training MSE (normalized units) = {loss:.6f}")

def predict_field(x_source):
    out, _ = forward(np.array([[x_source]]))
    return out[0]*Y_std + Y_mean

# ---------- test on an UNSEEN source position (interpolation, inside training range) ----------
x_test = 0.5
field_true = true_field(x_test)
field_pred = predict_field(x_test)
rel_err = np.linalg.norm(field_pred - field_true) / np.linalg.norm(field_true)
print(f"\nUnseen (interpolated) source position x={x_test}: relative L2 field error = {rel_err*100:.2f}%")

# ---------- test extrapolation OUTSIDE the training range ----------
x_extrap = 0.95
field_true_ex = true_field(x_extrap)
field_pred_ex = predict_field(x_extrap)
rel_err_ex = np.linalg.norm(field_pred_ex - field_true_ex) / np.linalg.norm(field_true_ex)
print(f"Extrapolated source position x={x_extrap} (outside training range [0.15,0.85]): "
      f"relative L2 field error = {rel_err_ex*100:.2f}%")
print("\nAs expected for any data-driven surrogate: interpolation error is small while")
print("extrapolation error is much larger -- always report the training envelope alongside")
print("any neural-operator benchmark result, exactly as the handbook's validation ladder demands.")
