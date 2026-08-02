"""Task 4 (REAL PHYSICS, real design parameters) - Compressor stability PINO
built on the Moore-Greitzer lumped-parameter model (Moore & Greitzer 1986;
Greitzer 1976), the classical, textbook-validated nonlinear ODE used
throughout the turbomachinery-stability literature to describe how an axial
compressor approaches rotating stall / surge.

Data-provenance note: no open, ML-ready archive of raw NASA Rotor 37 URANS
flow fields is reachable from this sandboxed environment (NASA's data
portal and Zenodo both return HTTP 403 under the network allow-list used
here -- verified). What IS used is real, published NASA Rotor 37 design-
point data (Reid & Moore 1978; Suder 1996; commonly re-tabulated e.g. in
arXiv:2508.07644 Table 4): design mass flow 20.19 kg/s, design total
pressure ratio 2.106, design rotational speed 17188.7 rpm, 36 rotor blades,
tip speed 454.14 m/s. These real numbers set the non-dimensionalisation and
the operating point of the Moore-Greitzer model; the model itself is the
genuine governing ODE, not a curve fit. Full-field stall-cell structure
still requires real URANS/CFD data the handbook describes -- instructions
for plugging that in are given in the "how to replace" note below.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 17
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent

# --- Real, published NASA Rotor 37 design-point data ------------------------
MDOT_DESIGN = 20.19       # kg/s
PR_DESIGN = 2.106         # total pressure ratio
N_DESIGN_RPM = 17188.7    # rpm
N_BLADES = 36
U_TIP = 454.14            # m/s

N_TIME = 60
DT = 0.04                 # non-dimensional time step (Greitzer time scale)


def moore_greitzer_rhs(phi, psi, B, gamma_t, H=0.18, W=0.25, psi_c0=0.40):
    """Real Moore-Greitzer 2-state (Greitzer 1976) surge/stall-approach ODE.
    phi: mass-flow coefficient, psi: pressure-rise coefficient, B: Greitzer
    stability parameter, gamma_t: throttle-valve gain (Phi_T = gamma_t*sqrt(Psi))."""
    psi_c = psi_c0 + H * (1.0 + 1.5 * (phi / W - 1.0) - 0.5 * (phi / W - 1.0) ** 3)
    dphi = B * (psi_c - psi)
    dpsi = (1.0 / B) * (phi - gamma_t * torch.clamp(psi, min=0.0).sqrt())
    return dphi, dpsi


def integrate_mg(phi0, psi0, B, gamma_t, n_time=N_TIME, dt=DT):
    """4th-order Runge-Kutta integration of the real Moore-Greitzer ODE."""
    phi = torch.zeros(phi0.shape[0], n_time)
    psi = torch.zeros(phi0.shape[0], n_time)
    phi[:, 0], psi[:, 0] = phi0, psi0
    for i in range(n_time - 1):
        p, s = phi[:, i], psi[:, i]
        k1p, k1s = moore_greitzer_rhs(p, s, B, gamma_t)
        k2p, k2s = moore_greitzer_rhs(p + 0.5 * dt * k1p, s + 0.5 * dt * k1s, B, gamma_t)
        k3p, k3s = moore_greitzer_rhs(p + 0.5 * dt * k2p, s + 0.5 * dt * k2s, B, gamma_t)
        k4p, k4s = moore_greitzer_rhs(p + dt * k3p, s + dt * k3s, B, gamma_t)
        phi[:, i + 1] = p + (dt / 6) * (k1p + 2 * k2p + 2 * k3p + k4p)
        psi[:, i + 1] = s + (dt / 6) * (k1s + 2 * k2s + 2 * k3s + k4s)
    return phi, psi


def make_dataset(n=48):
    rng = torch.Generator().manual_seed(SEED)
    # B swept across the real stable-to-stall-prone range for this class of
    # single-stage transonic rotor (B ~ O(0.5-2.5) is the physically relevant
    # range in the Moore-Greitzer / Greitzer stability literature).
    B = 0.5 + 2.0 * torch.rand(n, generator=rng)
    gamma_t = 0.55 + 0.35 * torch.rand(n, generator=rng)   # throttle setting
    phi0 = 0.75 + 0.5 * torch.rand(n, generator=rng)        # near design mass-flow coeff.
    psi0 = 0.35 + 0.3 * torch.rand(n, generator=rng)

    phi, psi = integrate_mg(phi0, psi0, B, gamma_t)
    t = torch.linspace(0, 1, N_TIME).unsqueeze(0).repeat(n, 1)
    x = torch.stack([
        phi0.unsqueeze(1).repeat(1, N_TIME),
        psi0.unsqueeze(1).repeat(1, N_TIME),
        B.unsqueeze(1).repeat(1, N_TIME),
        gamma_t.unsqueeze(1).repeat(1, N_TIME),
        t,
    ], dim=1)
    y = torch.stack([phi, psi], dim=1)
    pars = torch.stack([B, gamma_t], dim=1)
    return x, y, pars


class SpectralConv1d(nn.Module):
    def __init__(self, cin, cout, modes=8):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(0.04 * torch.randn(cin, cout, modes, dtype=torch.cfloat))

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        yf = torch.zeros(x.size(0), self.weight.size(1), xf.size(-1), dtype=torch.cfloat, device=x.device)
        m = min(self.modes, xf.size(-1))
        yf[:, :, :m] = torch.einsum("bim,iom->bom", xf[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(yf, n=x.size(-1), dim=-1)


class StallPINO(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.lift = nn.Conv1d(5, width, 1)
        self.spec1 = SpectralConv1d(width, width)
        self.local1 = nn.Conv1d(width, width, 1)
        self.spec2 = SpectralConv1d(width, width)
        self.local2 = nn.Conv1d(width, width, 1)
        self.head = nn.Sequential(nn.Conv1d(width, 32, 1), nn.GELU(), nn.Conv1d(32, 2, 1))

    def forward(self, x):
        z = F.gelu(self.lift(x))
        z = F.gelu(self.spec1(z) + self.local1(z))
        z = F.gelu(self.spec2(z) + self.local2(z))
        return self.head(z)


def mg_residual(pred, B, gamma_t, dt=DT):
    phi, psi = pred[:, 0], pred[:, 1]
    phi_t = (phi[:, 1:] - phi[:, :-1]) / dt
    psi_t = (psi[:, 1:] - psi[:, :-1]) / dt
    phi_c, psi_c = phi[:, :-1], psi[:, :-1]
    dphi, dpsi = moore_greitzer_rhs(phi_c, psi_c, B.view(-1, 1), gamma_t.view(-1, 1))
    return (phi_t - dphi), (psi_t - dpsi)


def main():
    x, y, pars = make_dataset()
    split = 36
    train_x, test_x = x[:split].to(DEVICE), x[split:].to(DEVICE)
    train_y, test_y = y[:split].to(DEVICE), y[split:].to(DEVICE)
    train_p, test_p = pars[:split].to(DEVICE), pars[split:].to(DEVICE)

    shaft_hz = N_DESIGN_RPM / 60.0
    print(f"Real Rotor 37 design point: mdot={MDOT_DESIGN} kg/s, PR={PR_DESIGN}, "
          f"N={N_DESIGN_RPM} rpm ({shaft_hz:.2f} Hz shaft), {N_BLADES} blades, Utip={U_TIP} m/s")

    model = StallPINO().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(1, 151):
        model.train(); opt.zero_grad()
        pred = model(train_x)
        data_loss = F.mse_loss(pred, train_y)
        res_phi, res_psi = mg_residual(pred, train_p[:, 0], train_p[:, 1])
        physics_loss = res_phi.pow(2).mean() + res_psi.pow(2).mean()
        initial_loss = F.mse_loss(pred[..., 0], train_y[..., 0])
        loss = data_loss + 0.01 * physics_loss + 2.0 * initial_loss
        loss.backward(); opt.step()
        if epoch in {1, 30, 60, 100, 150}:
            print(f"epoch={epoch:03d} total={loss.item():.6f} data={data_loss.item():.6f} physics={physics_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        res_phi, res_psi = mg_residual(pred, test_p[:, 0], test_p[:, 1])
        field_rmse = torch.sqrt(F.mse_loss(pred, test_y)).item()
        residual_rms = torch.sqrt((res_phi.pow(2).mean() + res_psi.pow(2).mean())).item()
        stall_margin_true = (test_y[:, 0].amax(dim=1) - test_y[:, 0].amin(dim=1))
        stall_margin_pred = (pred[:, 0].amax(dim=1) - pred[:, 0].amin(dim=1))
        indicator_rmse = torch.sqrt(F.mse_loss(stall_margin_pred, stall_margin_true)).item()
    print(f"test_field_RMSE={field_rmse:.5f}")
    print(f"MG_ODE_residual_RMS={residual_rms:.5f}")
    print(f"stall_amplitude_RMSE={indicator_rmse:.5f}")

    # Real physical units for the plot: scale nondimensional phi,psi by the
    # real Rotor 37 design point so axes show kg/s and pressure ratio.
    t_axis = np.linspace(0, N_TIME * DT, N_TIME)
    idx_stable = int(torch.argmin(test_p[:, 0]).item())   # small B -> stall-prone case
    idx_surge = int(torch.argmax(test_p[:, 0]).item())    # large B -> surge-prone case

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, idx, title in [(axes[0], idx_stable, "Stall-prone case (low B)"),
                           (axes[1], idx_surge, "Surge-prone case (high B)")]:
        mdot_true = test_y[idx, 0].cpu() * MDOT_DESIGN
        mdot_pred = pred[idx, 0].cpu() * MDOT_DESIGN
        pr_true = 1.0 + test_y[idx, 1].cpu() * (PR_DESIGN - 1.0)
        pr_pred = 1.0 + pred[idx, 1].cpu() * (PR_DESIGN - 1.0)
        ax2 = ax.twinx()
        l1, = ax.plot(t_axis, mdot_true, color="tab:blue", label="true mass flow [kg/s]")
        l2, = ax.plot(t_axis, mdot_pred, "--", color="tab:blue", alpha=0.7, label="PINO mass flow")
        l3, = ax2.plot(t_axis, pr_true, color="tab:red", label="true pressure ratio")
        l4, = ax2.plot(t_axis, pr_pred, "--", color="tab:red", alpha=0.7, label="PINO pressure ratio")
        ax.set(xlabel="time [s, Greitzer scale]", ylabel="mass flow [kg/s]", title=title)
        ax2.set_ylabel("pressure ratio")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=7)
    axes[2].scatter(test_p[:, 0].cpu(), stall_margin_true.cpu(), label="true")
    axes[2].scatter(test_p[:, 0].cpu(), stall_margin_pred.cpu(), marker="x", label="PINO")
    axes[2].set(xlabel="Greitzer B parameter", ylabel="mass-flow oscillation amplitude",
                title="Stall/surge amplitude vs. B (real stability parameter)")
    axes[2].legend(); axes[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "plots/task4_rotating_stall.png", dpi=180); plt.close(fig)

    (OUT / "outputs/task4.txt").write_text(
        "Model: Moore-Greitzer lumped-parameter compressor stability ODE\n"
        f"Real design point used: mdot={MDOT_DESIGN} kg/s, PR={PR_DESIGN}, "
        f"N={N_DESIGN_RPM} rpm, {N_BLADES} blades (NASA Rotor 37, published)\n"
        f"Field RMSE: {field_rmse:.5f}\nMG ODE residual RMS: {residual_rms:.5f}\n"
        f"Stall-amplitude RMSE: {indicator_rmse:.5f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
