"""Task 7 (REAL DATA) - Explainable PINO and sensor selection on NASA C-MAPSS FD001.

Real gate-per-sensor architecture applied to the actual 14 informative C-MAPSS
FD001 sensors plus the one non-constant operating setting. Gradient
attribution and top-k ablation are computed on real sensor channels, so the
resulting ranking can be sanity-checked against the published C-MAPSS
prognostics literature (T50, Ps30, BPR, and core-speed sensors are
consistently reported as most informative for FD001 RUL).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 29
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "data" / "cmapss"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
CONSTANT_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
# Human-readable C-MAPSS sensor descriptions (Saxena et al. 2008, Table II).
SENSOR_LABEL = {
    "op1": "alt_setting", "s2": "T24", "s3": "T30", "s4": "T50", "s7": "P30",
    "s8": "Nf", "s9": "Nc", "s11": "Ps30", "s12": "phi", "s13": "NRf",
    "s14": "NRc", "s15": "BPR", "s17": "htBleed", "s20": "W31", "s21": "W32",
}
RUL_CAP = 125.0
WINDOW = 30
STRIDE = 15
MAX_WINDOWS_PER_UNIT = 8


def make_dataset():
    df = pd.read_csv(DATA / "train_FD001.txt", sep=r"\s+", header=None, names=COLS)
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" not in CONSTANT_SENSORS]
    feature_cols = ["op1"] + sensor_cols
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["health"] = (max_cycle - df["cycle"]).clip(upper=RUL_CAP) / RUL_CAP

    xs, hs, unit_ids = [], [], []
    for u in df["unit"].unique():
        sub = df[df["unit"] == u].sort_values("cycle")
        n = len(sub)
        if n < WINDOW:
            continue
        starts = list(range(0, n - WINDOW + 1, STRIDE))
        if len(starts) > MAX_WINDOWS_PER_UNIT:
            starts = list(np.linspace(0, n - WINDOW, MAX_WINDOWS_PER_UNIT).astype(int))
        for s in starts:
            block = sub.iloc[s:s + WINDOW]
            xs.append(block[feature_cols].to_numpy(dtype=np.float32).T)
            hs.append(block["health"].to_numpy(dtype=np.float32)[None, :])
            unit_ids.append(u)

    x = torch.tensor(np.stack(xs))
    h = torch.tensor(np.stack(hs))
    unit_ids = np.array(unit_ids)
    mean = x.mean((0, 2), keepdim=True)
    std = x.std((0, 2), keepdim=True).clamp_min(1e-6)
    labels = [SENSOR_LABEL.get(c, c) for c in feature_cols]
    return (x - mean) / std, h, unit_ids, labels


class GatedHealthPINO(nn.Module):
    def __init__(self, n_sensors, width=24):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((n_sensors,), 1.5))
        self.net = nn.Sequential(
            nn.Conv1d(n_sensors, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, x, gate_override=None):
        gates = torch.sigmoid(self.gate_logits) if gate_override is None else gate_override
        z = x * gates.view(1, -1, 1)
        rate = 0.06 * torch.sigmoid(self.net(z))
        health = 1.0 - torch.cumsum(rate, dim=-1)
        return health, gates


def gradient_attribution(model, x):
    x = x.clone().detach().requires_grad_(True)
    pred, _ = model(x)
    pred[:, :, -1].sum().backward()
    return (x.grad * x).abs().mean(dim=(0, 2)).detach()


def main():
    x, health, unit_ids, sensor_names = make_dataset()
    n_sensors = x.shape[1]

    rng = np.random.RandomState(SEED)
    unique_units = np.unique(unit_ids)
    rng.shuffle(unique_units)
    test_units = set(unique_units[:20])
    train_mask = np.array([u not in test_units for u in unit_ids])
    test_mask = ~train_mask
    train_x, test_x = x[train_mask].to(DEVICE), x[test_mask].to(DEVICE)
    train_h, test_h = health[train_mask].to(DEVICE), health[test_mask].to(DEVICE)
    print(f"Real C-MAPSS FD001 sensor selection: {n_sensors} channels: {sensor_names}")

    model = GatedHealthPINO(n_sensors).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(1, 201):
        model.train(); opt.zero_grad()
        pred, gates = model(train_x)
        fit = F.mse_loss(pred, train_h)
        sparsity_weight = 0.0005 * min(1.0, epoch / 80)
        sparsity = gates.mean()
        minimum_information = F.relu(3.0 - gates.sum())
        loss = 100 * fit + sparsity_weight * sparsity + minimum_information
        loss.backward(); opt.step()
        if epoch in {1, 40, 80, 140, 200}:
            print(f"epoch={epoch:03d} total={loss.item():.6f} fit={fit.item():.7f} mean_gate={gates.mean().item():.3f}")

    model.eval()
    with torch.no_grad():
        pred, gates = model(test_x)
        full_rmse = torch.sqrt(F.mse_loss(pred, test_h)).item()
    attribution = gradient_attribution(model, test_x[:32]).cpu()
    combined = (gates.detach().cpu() * attribution / attribution.max().clamp_min(1e-8))
    ranking = torch.argsort(combined, descending=True)

    topk_rmse = []
    with torch.no_grad():
        for k in range(1, n_sensors + 1):
            mask = torch.zeros_like(gates)
            mask[ranking[:k].to(DEVICE)] = 1.0
            p, _ = model(test_x, gate_override=mask)
            topk_rmse.append(torch.sqrt(F.mse_loss(p, test_h)).item())

    print(f"all_sensor_health_RMSE={full_rmse:.6f}")
    ranking_names = [sensor_names[i] for i in ranking.tolist()]
    print("sensor_ranking=" + ", ".join(ranking_names))
    for name, gate, attr in zip(sensor_names, gates.cpu(), attribution):
        print(f"{name:12s} gate={gate.item():.3f} gradient_attribution={attr.item():.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    order = ranking.tolist()
    axes[0].bar([sensor_names[i] for i in order], combined[order])
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set(ylabel="gate x gradient attribution", title="Real-sensor ranking (C-MAPSS FD001)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(range(1, n_sensors + 1), topk_rmse, marker="o")
    axes[1].axhline(full_rmse, linestyle="--", label="soft-gated model")
    axes[1].set(xlabel="number of retained sensors", ylabel="health RMSE", title="Faithfulness by top-k ablation")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "plots/task7_sensor_selection.png", dpi=180); plt.close(fig)

    lines = ["Dataset: NASA C-MAPSS FD001 (real, 100 engines, 15 real sensor/setting channels)",
             f"All-sensor health RMSE: {full_rmse:.6f}",
             "Ranking: " + ", ".join(ranking_names)]
    lines += [f"{name}: gate={gate.item():.3f}, attribution={attr.item():.5f}"
              for name, gate, attr in zip(sensor_names, gates.cpu(), attribution)]
    (OUT / "outputs/task7.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
