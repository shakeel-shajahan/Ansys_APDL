"""Task 6 (REAL DATA) - Separate real component faults from injected sensor
faults, using the real CWRU Bearing Data Center vibration signals.

Component-fault labels (healthy / inner race / outer race / ball) are REAL
seeded bearing defects from CWRU. Sensor-fault labels (bias / drift /
dropout) are synthetically injected on top of the real DE/FE channels --
exactly the augmentation strategy the handbook itself recommends for
N-CMAPSS ("augment it with synthetic sensor degradation"), applied here to
CWRU because the primary N-CMAPSS/Paderborn archives are not reachable from
this sandboxed environment (see the Data Provenance appendix).
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 23
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "data" / "cwru"

FS = 12000.0
WINDOW = 2048
STRIDE = 1024
COMPONENT_NAMES = ["healthy", "inner race", "outer race", "ball"]
SENSOR_NAMES = ["none", "bias", "drift", "dropout"]


def load_channel_pair(path):
    d = np.load(path)
    de = d["DE"].astype(np.float32).ravel()
    fe = d["FE"].astype(np.float32).ravel()
    n = min(len(de), len(fe))
    return np.stack([de[:n], fe[:n]])


def windows_from_signal(sig, window=WINDOW, stride=STRIDE):
    n = sig.shape[-1]
    return np.stack([sig[:, s:s + window] for s in range(0, n - window + 1, stride)])


def inject_sensor_fault(window, fault_class, rng):
    """Apply a synthetic sensor fault to one real channel of a real window."""
    w = window.copy()
    channel = rng.randint(0, 2)
    n = w.shape[-1]
    if fault_class == 1:  # bias
        w[channel] += 3.0 * w[channel].std()
    elif fault_class == 2:  # drift
        w[channel] += np.linspace(0.0, 4.0 * w[channel].std(), n)
    elif fault_class == 3:  # dropout
        start = rng.randint(n // 4, n // 2)
        w[channel, start:start + n // 6] = 0.0
    return w


def make_dataset():
    raw = {
        0: load_channel_pair(DATA / "1797_Normal.npz"),
        1: load_channel_pair(DATA / "1797_IR_7_DE12.npz"),
        2: load_channel_pair(DATA / "1797_OR@6_7_DE12.npz"),
        3: load_channel_pair(DATA / "1797_B_7_DE12.npz"),
    }
    windows = {c: windows_from_signal(sig) for c, sig in raw.items()}
    rng = np.random.RandomState(SEED)

    n_per_class = min(len(w) for w in windows.values())
    observed, clean, component, sensor = [], [], [], []
    for c in range(4):
        idx = rng.choice(len(windows[c]), size=n_per_class, replace=False)
        for j, i in enumerate(idx):
            clean_w = windows[c][i]
            s_class = j % 4
            obs_w = inject_sensor_fault(clean_w, s_class, rng)
            observed.append(obs_w); clean.append(clean_w)
            component.append(c); sensor.append(s_class)

    observed = np.stack(observed).astype(np.float32)
    clean = np.stack(clean).astype(np.float32)
    order = rng.permutation(len(observed))
    observed, clean = observed[order], clean[order]
    component = np.array(component)[order]
    sensor = np.array(sensor)[order]

    observed_t = torch.tensor(observed)
    clean_t = torch.tensor(clean)
    split = int(0.75 * len(observed_t))
    mean = observed_t[:split].mean((0, 2), keepdim=True)
    std = observed_t[:split].std((0, 2), keepdim=True).clamp_min(1e-6)
    observed_t = (observed_t - mean) / std
    clean_t = (clean_t - mean) / std
    return (observed_t, clean_t, torch.tensor(component), torch.tensor(sensor),
            mean, std, split)


class DisentangledFaultNet(nn.Module):
    def __init__(self, channels=2, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, width, 15, padding=7, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 9, padding=4, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
        )
        self.component_head = nn.Linear(width, 4)
        self.sensor_head = nn.Linear(width, 4)
        self.clean_head = nn.Sequential(
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(width, channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        pooled = z.mean(dim=-1)
        return self.component_head(pooled), self.sensor_head(pooled), self.clean_head(z)


def confusion_matrix(true, pred, n_class=4):
    matrix = torch.zeros(n_class, n_class, dtype=torch.int64)
    for t, p in zip(true, pred):
        matrix[int(t), int(p)] += 1
    return matrix


def main():
    x, clean, c_label, s_label, mean, std, split = make_dataset()
    train = [z[:split].to(DEVICE) for z in (x, clean, c_label, s_label)]
    test = [z[split:].to(DEVICE) for z in (x, clean, c_label, s_label)]
    print(f"Real CWRU + injected sensor faults: {split} train / {len(x) - split} test windows")

    model = DisentangledFaultNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    for epoch in range(1, 101):
        model.train(); opt.zero_grad()
        c_logit, s_logit, clean_pred = model(train[0])
        component_loss = F.cross_entropy(c_logit, train[2])
        sensor_loss = F.cross_entropy(s_logit, train[3])
        reconstruction_loss = F.mse_loss(clean_pred, train[1])
        loss = component_loss + sensor_loss + reconstruction_loss
        loss.backward(); opt.step()
        if epoch in {1, 25, 50, 75, 100}:
            print(f"epoch={epoch:03d} total={loss.item():.4f} component={component_loss.item():.4f} sensor={sensor_loss.item():.4f} reconstruction={reconstruction_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        c_logit, s_logit, clean_pred = model(test[0])
        c_pred, s_pred = c_logit.argmax(1), s_logit.argmax(1)
        c_acc = (c_pred == test[2]).float().mean().item()
        s_acc = (s_pred == test[3]).float().mean().item()
        clean_rmse = torch.sqrt(F.mse_loss(clean_pred, test[1])).item()
        cm_c = confusion_matrix(test[2].cpu(), c_pred.cpu())
        cm_s = confusion_matrix(test[3].cpu(), s_pred.cpu())
    print(f"component_fault_accuracy={c_acc:.3f}")
    print(f"sensor_fault_accuracy={s_acc:.3f}")
    print(f"clean_signal_RMSE={clean_rmse:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    im0 = axes[0].imshow(cm_c, cmap="Greys")
    axes[0].set(title="Component-fault confusion\n(real CWRU labels)", xlabel="predicted", ylabel="true",
                xticks=range(4), yticks=range(4))
    axes[0].set_xticklabels(COMPONENT_NAMES, rotation=45, ha="right")
    axes[0].set_yticklabels(COMPONENT_NAMES)
    for i in range(4):
        for j in range(4): axes[0].text(j, i, int(cm_c[i, j]), ha="center", va="center")
    im1 = axes[1].imshow(cm_s, cmap="Greys")
    axes[1].set(title="Sensor-fault confusion\n(synthetic injection)", xlabel="predicted", ylabel="true",
                xticks=range(4), yticks=range(4))
    axes[1].set_xticklabels(SENSOR_NAMES, rotation=45, ha="right")
    axes[1].set_yticklabels(SENSOR_NAMES)
    for i in range(4):
        for j in range(4): axes[1].text(j, i, int(cm_s[i, j]), ha="center", va="center")
    # pick a test window that actually had a sensor fault injected
    fault_positions = (test[3] != 0).nonzero(as_tuple=True)[0]
    idx = int(fault_positions[0]) if len(fault_positions) else 0
    axes[2].plot(test[0][idx, 0].cpu(), label="observed DE (with sensor fault)")
    axes[2].plot(test[1][idx, 0].cpu(), label="clean DE target")
    axes[2].plot(clean_pred[idx, 0].detach().cpu(), "--", label="reconstructed clean")
    axes[2].set(title="Removing an injected sensor fault\n(real vibration signal)", xlabel="sample", ylabel="normalised amplitude")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "plots/task6_fault_disentanglement.png", dpi=180); plt.close(fig)

    (OUT / "outputs/task6.txt").write_text(
        "Dataset: CWRU Bearing Data Center (real component faults) + synthetic\n"
        "sensor-fault injection (bias/drift/dropout) on real DE/FE channels\n"
        f"Component-fault accuracy: {c_acc:.3f}\nSensor-fault accuracy: {s_acc:.3f}\n"
        f"Clean-signal RMSE: {clean_rmse:.4f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
