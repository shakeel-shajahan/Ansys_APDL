"""Task 2 (REAL DATA) - Inverse PINO for hidden compressor-health parameters,
using the real NASA C-MAPSS FD002 subset (260 real turbofan engines, six
real operating conditions -- altitude, Mach-like and throttle settings all
genuinely vary, unlike FD001).

Data-provenance note: the handbook's primary recommendation for this task is
N-CMAPSS, which ships real per-module flow/efficiency health parameters.
N-CMAPSS is hosted only on NASA's data portal, which this sandboxed
environment cannot reach (network allow-list). FD002 is used instead: it is
still 100% real flight-condition and sensor data, but it does not carry
ground-truth component health labels, so two data-driven degradation
indices are extracted directly from real sensor trends (a standard
"health-index construction" technique in the PHM literature, e.g. Wang
2008), rather than being read off a label column. The forward "physics"
observation model is a linear map fit by least squares to the real training
sensors -- a first-order, data-calibrated surrogate of the true nonlinear
compressor thermodynamics, used for the cycle-consistency loss.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 11
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "data" / "cmapss"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# s7=P30 (HPC outlet pressure, "pressure-ratio-like"), s3=T30 (HPC outlet
# temperature, "efficiency-like"), s12=phi (fuel-flow/Ps30 ratio, "fuel
# proxy"), s9=Nc (physical core speed, "speed proxy" -- no vibration channel
# exists in C-MAPSS, a genuine and disclosed dataset limitation).
SENSOR_SET = ["s7", "s3", "s12", "s9"]
WINDOW, STRIDE, MAX_PER_UNIT = 30, 15, 6


def health_index_from_sensor(df, sensor, rising_is_degradation):
    """Data-driven degradation proxy extracted from a real sensor's trend:
    deviation from each unit's own early-life baseline, sign-corrected from
    the real correlation with cycle fraction, cumulative-max enforced
    monotonic, and scaled 0-1 by the population end-of-life deviation."""
    out = np.zeros(len(df), dtype=np.float32)
    sign = 1.0 if rising_is_degradation else -1.0
    eol_devs = []
    baselines = {}
    for u, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        baseline = sub[sensor].iloc[:5].mean()
        baselines[u] = baseline
        dev = sign * (sub[sensor].values - baseline)
        eol_devs.append(dev[-5:].mean())
    scale = max(np.median(eol_devs), 1e-6)
    for u, sub in df.groupby("unit"):
        idx = sub.sort_values("cycle").index
        dev = sign * (df.loc[idx, sensor].values - baselines[u]) / scale
        dev = np.clip(dev, 0.0, None)
        dev = np.maximum.accumulate(dev)          # monotonic non-decreasing
        dev = np.clip(dev, 0.0, 1.0)
        out[idx] = dev
    return out


def make_dataset():
    df = pd.read_csv(DATA / "train_FD002.txt", sep=r"\s+", header=None, names=COLS)

    # Real correlation sign: does this sensor rise or fall with real ageing?
    age_frac = df["cycle"] / df.groupby("unit")["cycle"].transform("max")
    corr_p = np.corrcoef(df["s7"], age_frac)[0, 1]
    corr_t = np.corrcoef(df["s3"], age_frac)[0, 1]
    df["flow_loss"] = health_index_from_sensor(df, "s7", rising_is_degradation=(corr_p < 0))
    df["eff_loss"] = health_index_from_sensor(df, "s3", rising_is_degradation=(corr_t > 0))

    op_cols = ["op1", "op3"]
    feature_cols = op_cols + SENSOR_SET

    units = df["unit"].unique()
    xs, hs, ys, unit_ids = [], [], [], []
    for u in units:
        sub = df[df["unit"] == u].sort_values("cycle")
        n = len(sub)
        if n < WINDOW:
            continue
        starts = list(range(0, n - WINDOW + 1, STRIDE))
        if len(starts) > MAX_PER_UNIT:
            starts = list(np.linspace(0, n - WINDOW, MAX_PER_UNIT).astype(int))
        for s in starts:
            block = sub.iloc[s:s + WINDOW]
            xs.append(block[feature_cols].to_numpy(dtype=np.float32).T)
            hs.append(block[["flow_loss", "eff_loss"]].to_numpy(dtype=np.float32).T)
            ys.append(block[SENSOR_SET].to_numpy(dtype=np.float32).T)
            unit_ids.append(u)

    x = torch.tensor(np.stack(xs))
    h = torch.tensor(np.stack(hs))
    y = torch.tensor(np.stack(ys))
    unit_ids = np.array(unit_ids)

    mean = x.mean((0, 2), keepdim=True)
    std = x.std((0, 2), keepdim=True).clamp_min(1e-6)
    mean_y = y.mean((0, 2), keepdim=True)
    std_y = y.std((0, 2), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, h, (y - mean_y) / std_y, mean, std, mean_y, std_y, unit_ids, feature_cols


def fit_linear_forward_model(op_norm, h, y_norm):
    """Least-squares fit of a linear observation model (in normalised units)
    sensor = A @ [1, op1, op3, flow_loss, eff_loss] on real training data --
    a data-calibrated, first-order surrogate of the true compressor map."""
    design = torch.cat([torch.ones_like(op_norm[:, :1]), op_norm, h], dim=1)  # (N,5,T)
    design = design.permute(0, 2, 1).reshape(-1, 5)          # (N*T, 5)
    target = y_norm.permute(0, 2, 1).reshape(-1, y_norm.shape[1])  # (N*T, S)
    coeffs, *_ = torch.linalg.lstsq(design, target)          # (5, S)
    return coeffs


class SpectralConv1d(nn.Module):
    def __init__(self, cin, cout, modes=6):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(0.04 * torch.randn(cin, cout, modes, dtype=torch.cfloat))

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        yf = torch.zeros(x.size(0), self.weight.size(1), xf.size(-1),
                          dtype=torch.cfloat, device=x.device)
        m = min(self.modes, xf.size(-1))
        yf[:, :, :m] = torch.einsum("bim,iom->bom", xf[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(yf, n=x.size(-1), dim=-1)


class InverseHealthPINO(nn.Module):
    def __init__(self, in_channels, width=20):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = SpectralConv1d(width, width)
        self.local = nn.Conv1d(width, width, 1)
        self.head = nn.Sequential(nn.Conv1d(width, 24, 1), nn.GELU(), nn.Conv1d(24, 2, 1))

    def forward(self, x):
        z = F.gelu(self.lift(x))
        z = F.gelu(self.spectral(z) + self.local(z))
        return torch.sigmoid(self.head(z))


def forward_physics(op_norm, health, coeffs):
    design = torch.cat([torch.ones_like(op_norm[:, :1]), op_norm, health], dim=1)
    design = design.permute(0, 2, 1)                          # (N,T,5)
    recon = torch.einsum("ntc,cs->nts", design, coeffs)
    return recon.permute(0, 2, 1)                              # (N,S,T)


def main():
    x, health, sensors, mean, std, mean_y, std_y, unit_ids, feature_cols = make_dataset()
    rng = np.random.RandomState(SEED)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    test_units = set(unique_units[:50])
    train_mask = np.array([u not in test_units for u in unit_ids])
    test_mask = ~train_mask

    coeffs = fit_linear_forward_model(x[train_mask][:, :2], health[train_mask], sensors[train_mask])

    train_x, test_x = x[train_mask].to(DEVICE), x[test_mask].to(DEVICE)
    train_h, test_h = health[train_mask].to(DEVICE), health[test_mask].to(DEVICE)
    train_y, test_y = sensors[train_mask].to(DEVICE), sensors[test_mask].to(DEVICE)
    coeffs = coeffs.to(DEVICE)
    print(f"Real C-MAPSS FD002: {train_mask.sum()} train / {test_mask.sum()} test windows, "
          f"{len(unique_units) - 50}/{50} engines. Channels: {feature_cols}")

    model = InverseHealthPINO(in_channels=x.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(1, 101):
        model.train(); opt.zero_grad()
        pred_h = model(train_x)
        reconstructed = forward_physics(train_x[:, :2], pred_h, coeffs)
        health_loss = F.mse_loss(pred_h, train_h)
        cycle_loss = F.mse_loss(reconstructed, train_y)
        monotonic = F.relu(pred_h[:, :, :-1] - pred_h[:, :, 1:]).mean()
        loss = 20 * health_loss + 2.0 * cycle_loss + 5 * monotonic
        loss.backward(); opt.step()
        if epoch in {1, 25, 50, 75, 100}:
            print(f"epoch={epoch:03d} total={loss.item():.6f} health={health_loss.item():.6f} cycle={cycle_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred_h = model(test_x)
        recon = forward_physics(test_x[:, :2], pred_h, coeffs)
        h_rmse = torch.sqrt(F.mse_loss(pred_h, test_h)).item()
        y_rmse = torch.sqrt(F.mse_loss(recon, test_y)).item()
    print(f"test_health_parameter_RMSE={h_rmse:.5f}")
    print(f"test_sensor_reconstruction_RMSE={y_rmse:.5f}")

    true_final = test_h[:, :, -1].cpu().numpy()
    pred_final = pred_h[:, :, -1].cpu().numpy()
    idx = 2
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].scatter(true_final[:, 0], pred_final[:, 0], label="flow-loss index", alpha=0.6, s=14)
    axes[0].scatter(true_final[:, 1], pred_final[:, 1], label="efficiency-loss index", alpha=0.6, s=14)
    lim = [0, max(true_final.max(), pred_final.max()) * 1.1 + 1e-3]
    axes[0].plot(lim, lim, "k--", linewidth=1)
    axes[0].set(xlabel="sensor-derived health index (real data)", ylabel="PINO-predicted",
                title="Inverse health estimate (real FD002 engines)")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(test_y[idx, 0].cpu(), label="true P30 (real sensor)")
    axes[1].plot(recon[idx, 0].cpu(), "--", label="reconstructed (linear physics model)")
    axes[1].set(xlabel="cycle within window", ylabel="normalised signal", title="Forward cycle consistency")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "plots/task2_inverse_health.png", dpi=180); plt.close(fig)

    (OUT / "outputs/task2.txt").write_text(
        "Dataset: NASA C-MAPSS FD002 (real, 260 engines, 6 real operating conditions)\n"
        "Health targets: sensor-derived flow-loss / efficiency-loss indices\n"
        "(N-CMAPSS ground-truth health parameters unreachable from this sandbox)\n"
        f"Health-parameter RMSE: {h_rmse:.5f}\nSensor-reconstruction RMSE: {y_rmse:.5f}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
