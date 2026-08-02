"""Task 1 (REAL DATA) - Temporal PINO for health and RUL on NASA C-MAPSS FD001.

This is the same physics-informed temporal operator as the beginner smoke
test, but ``make_dataset`` now loads the actual NASA C-MAPSS FD001 turbofan
degradation trajectories (100 real engine units, real sensor channels)
instead of synthetic curves. Data provenance: fetched from the public GitHub
mirror of the official NASA PCoE C-MAPSS release (see README in this folder).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 7
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "data" / "cmapss"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# Sensors with near-zero variance in FD001 (constant instrument readings at
# the single sea-level operating condition); dropped exactly as literature
# recommends for FD001 (Saxena et al. 2008; Heimes 2008).
CONSTANT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
RUL_CAP = 125.0   # standard piecewise-linear RUL cap used across C-MAPSS studies
WINDOW = 30       # cycles per input window
STRIDE = 15
MAX_WINDOWS_PER_UNIT = 8


def load_cmapss(split="train"):
    path = DATA / f"{split}_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df


def make_dataset():
    df = load_cmapss("train")
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" not in CONSTANT_SENSORS]
    feature_cols = ["op1"] + sensor_cols  # op2/op3 constant in FD001, dropped too

    # Real degradation label: piecewise-linear RUL, standard C-MAPSS practice.
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).clip(upper=RUL_CAP)
    df["health"] = rul / RUL_CAP          # 1.0 = healthy, 0.0 = at/near failure
    df["rul_cycles"] = rul

    units = df["unit"].unique()
    windows_x, windows_h, windows_rul, unit_ids = [], [], [], []
    for u in units:
        sub = df[df["unit"] == u].sort_values("cycle")
        n = len(sub)
        if n < WINDOW:
            continue
        starts = list(range(0, n - WINDOW + 1, STRIDE))
        if len(starts) > MAX_WINDOWS_PER_UNIT:
            starts = list(np.linspace(0, n - WINDOW, MAX_WINDOWS_PER_UNIT).astype(int))
        for s in starts:
            block = sub.iloc[s:s + WINDOW]
            x = block[feature_cols].to_numpy(dtype=np.float32).T   # (C, WINDOW)
            h = block["health"].to_numpy(dtype=np.float32)[None, :]
            windows_x.append(x)
            windows_h.append(h)
            windows_rul.append(block["rul_cycles"].to_numpy(dtype=np.float32)[-1])
            unit_ids.append(u)

    x = torch.tensor(np.stack(windows_x))
    health = torch.tensor(np.stack(windows_h))
    rul = torch.tensor(np.array(windows_rul)).view(-1, 1)
    unit_ids = np.array(unit_ids)

    # rate target consistent with the health cumulative-sum construction
    rate = torch.zeros_like(health)
    rate[:, :, 1:] = (health[:, :, :-1] - health[:, :, 1:]).clamp(min=0.0)
    rate[:, :, 0:1] = rate[:, :, 1:2]

    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    return x, health, rate, rul, unit_ids, feature_cols


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.modes = modes

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            x.size(0), self.weight.size(1), x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)


class TemporalPINO(nn.Module):
    def __init__(self, in_channels, width=16, modes=6):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spec1 = SpectralConv1d(width, width, modes)
        self.skip1 = nn.Conv1d(width, width, 1)
        self.spec2 = SpectralConv1d(width, width, modes)
        self.skip2 = nn.Conv1d(width, width, 1)
        self.decode_rate = nn.Sequential(
            nn.Conv1d(width, 24, 1), nn.GELU(), nn.Conv1d(24, 1, 1)
        )

    def forward(self, x):
        z = self.lift(x)
        z = F.gelu(self.spec1(z) + self.skip1(z))
        z = F.gelu(self.spec2(z) + self.skip2(z))
        rate = 0.06 * torch.sigmoid(self.decode_rate(z))
        health = 1.0 - torch.cumsum(rate, dim=-1)
        rul = ((health[:, :, -1]).clamp(min=0.0) * RUL_CAP)
        return health, rate, rul


def loss_terms(pred_h, pred_r, pred_rul, true_h, true_r, true_rul):
    health_fit = F.mse_loss(pred_h, true_h)
    rate_fit = F.mse_loss(pred_r, true_r)
    rul_fit = F.mse_loss(pred_rul / RUL_CAP, true_rul / RUL_CAP)
    dh = pred_h[:, :, 1:] - pred_h[:, :, :-1]
    ode = (dh + pred_r[:, :, 1:]).pow(2).mean()
    monotonic = F.relu(dh).mean()
    total = 20.0 * health_fit + 10.0 * rate_fit + rul_fit + 50.0 * ode
    return total, health_fit + rul_fit, ode, monotonic


def main():
    x, health, rate, rul, unit_ids, feature_cols = make_dataset()
    n_channels = x.shape[1]

    # Leakage-free split: hold out entire engine units, never overlapping windows.
    rng = np.random.RandomState(SEED)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    n_test_units = 20
    test_units = set(unique_units[:n_test_units])
    train_mask = np.array([u not in test_units for u in unit_ids])
    test_mask = ~train_mask

    train = [t[train_mask].to(DEVICE) for t in (x, health, rate, rul)]
    test = [t[test_mask].to(DEVICE) for t in (x, health, rate, rul)]
    print(f"Real C-MAPSS FD001: {n_channels} channels, "
          f"{train_mask.sum()} train windows / {test_mask.sum()} test windows "
          f"({len(unique_units) - n_test_units} train engines / {n_test_units} test engines)")

    model = TemporalPINO(in_channels=n_channels).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []

    n_epochs = 150
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        ph, pr, prul = model(train[0])
        total, data, ode, mono = loss_terms(ph, pr, prul, *train[1:])
        total.backward()
        optimiser.step()
        history.append(float(total.detach()))
        if epoch in {1, 25, 50, 100, 150}:
            print(
                f"epoch={epoch:03d} total={total.item():.5f} "
                f"data={data.item():.5f} physics={ode.item():.6f} "
                f"monotonic={mono.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        ph, pr, prul = model(test[0])
        health_rmse = torch.sqrt(F.mse_loss(ph, test[1])).item()
        rul_rmse = torch.sqrt(F.mse_loss(prul, test[3])).item()
        violation = (ph[:, :, 1:] > ph[:, :, :-1]).float().mean().item() * 100
    print(f"test_health_RMSE={health_rmse:.4f}")
    print(f"test_RUL_RMSE={rul_rmse:.3f} cycles (real FD001 test engines)")
    print(f"monotonicity_violations={violation:.2f}%")

    idx = 3
    t = np.arange(ph.size(-1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(t, test[1][idx, 0].cpu(), label="true health (RUL-derived)")
    axes[0].plot(t, ph[idx, 0].cpu(), "--", label="PINO health")
    axes[0].set(xlabel="cycle within window", ylabel="health", title="Health trajectory (real FD001 engine)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(history)
    axes[1].set(xlabel="epoch", ylabel="total loss", title="Training convergence")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "plots/task1_health_rul.png", dpi=180)
    plt.close(fig)

    (OUT / "outputs/task1.txt").write_text(
        "Dataset: NASA C-MAPSS FD001 (real, 100 engines)\n"
        f"Channels used ({n_channels}): {', '.join(feature_cols)}\n"
        f"Train/test split: {len(unique_units) - n_test_units}/{n_test_units} engine units (leakage-free)\n"
        f"Health RMSE: {health_rmse:.4f}\n"
        f"RUL RMSE: {rul_rmse:.3f} cycles\n"
        f"Monotonicity violations: {violation:.2f}%\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
