"""Task 3 (REAL DATA) - Self-supervised multimodal anomaly detector on the
real Case Western Reserve University (CWRU) Bearing Data Center dataset.

Data provenance: real drive-end (DE) and fan-end (FE) accelerometer signals
at 1797 RPM, 12 kHz sampling, for a healthy bearing and three real seeded
faults (inner race, outer race, ball), fetched from the public GitHub mirror
of the official CWRU Bearing Data Center .mat files (converted to .npz).
CWRU itself is a rotating-machinery rig rather than an axial-compressor
stage -- exactly the transfer-learning role the handbook assigns to it,
since no open, ML-ready axial-compressor vibration archive exists.

Physics-informed component: real SKF6205 drive-end bearing geometry gives
the true bearing characteristic fault frequencies (BPFO, BPFI, BSF) at the
real 1797 RPM shaft speed. The autoencoder is penalised for not preserving
narrowband energy at the real shaft rotational frequency, and the anomaly
score is augmented with real fault-frequency-band spectral energy -- an
actual order-frequency physical consistency check, not a synthetic one.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 13
torch.set_num_threads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "data" / "cwru"

FS = 12000.0          # real CWRU drive-end sampling rate [Hz]
RPM = 1797.0           # real CWRU test motor speed for this file set
WINDOW = 2048
STRIDE = 1024

# Real SKF6205 drive-end bearing geometry (CWRU Bearing Data Center specs).
N_BALLS, BALL_D, PITCH_D, CONTACT_ANGLE = 9, 0.3126, 1.537, 0.0


def bearing_fault_frequencies(rpm=RPM):
    fr = rpm / 60.0
    ratio = (BALL_D / PITCH_D) * np.cos(CONTACT_ANGLE)
    bpfo = N_BALLS / 2 * fr * (1 - ratio)
    bpfi = N_BALLS / 2 * fr * (1 + ratio)
    bsf = (PITCH_D / (2 * BALL_D)) * fr * (1 - ratio ** 2)
    return fr, bpfo, bpfi, bsf


def load_channel_pair(path):
    d = np.load(path)
    de = d["DE"].astype(np.float32).ravel()
    fe = d["FE"].astype(np.float32).ravel()
    n = min(len(de), len(fe))
    return np.stack([de[:n], fe[:n]])   # (2, n)


def windows_from_signal(sig, window=WINDOW, stride=STRIDE):
    n = sig.shape[-1]
    out = []
    for start in range(0, n - window + 1, stride):
        out.append(sig[:, start:start + window])
    return np.stack(out) if out else np.zeros((0, sig.shape[0], window), dtype=np.float32)


def make_dataset():
    normal = load_channel_pair(DATA / "1797_Normal.npz")
    inner = load_channel_pair(DATA / "1797_IR_7_DE12.npz")
    outer = load_channel_pair(DATA / "1797_OR@6_7_DE12.npz")
    ball = load_channel_pair(DATA / "1797_B_7_DE12.npz")

    w_normal = windows_from_signal(normal)
    w_inner = windows_from_signal(inner)
    w_outer = windows_from_signal(outer)
    w_ball = windows_from_signal(ball)

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(w_normal))
    n_train = int(0.7 * len(w_normal))
    train = torch.tensor(w_normal[idx[:n_train]])
    test_normal = torch.tensor(w_normal[idx[n_train:]])

    def sample(arr, k):
        pick = rng.choice(len(arr), size=min(k, len(arr)), replace=False)
        return torch.tensor(arr[pick])

    k = min(len(test_normal), len(w_inner), len(w_outer), len(w_ball))
    test_normal = test_normal[:k]
    test_inner = sample(w_inner, k)
    test_outer = sample(w_outer, k)
    test_ball = sample(w_ball, k)

    mean = train.mean((0, 2), keepdim=True)
    std = train.std((0, 2), keepdim=True).clamp_min(1e-6)
    train = (train - mean) / std
    test_normal = (test_normal - mean) / std
    test_inner = (test_inner - mean) / std
    test_outer = (test_outer - mean) / std
    test_ball = (test_ball - mean) / std

    test = torch.cat([test_normal, test_inner, test_outer, test_ball])
    labels = torch.cat([
        torch.zeros(k), torch.ones(k), 2 * torch.ones(k), 3 * torch.ones(k)
    ])
    return train, test, labels, mean, std, k


def random_mask(x):
    masked = x.clone()
    for b in range(x.size(0)):
        start = torch.randint(0, x.size(-1) - 200, (1,)).item()
        masked[b, :, start:start + 200] = 0
        if torch.rand(1) < 0.4:
            masked[b, torch.randint(0, x.size(1), (1,)).item()] = 0
    return masked


class MaskedAutoencoder(nn.Module):
    def __init__(self, channels=2, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, width, 15, padding=7, stride=2), nn.GELU(),
            nn.Conv1d(width, width, 9, padding=4), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(width, channels, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def band_energy(x, fs, f_lo, f_hi):
    """Differentiable narrowband spectral energy, normalised by total signal
    energy so the physics term stays on the same numerical scale as the
    reconstruction loss regardless of raw FFT magnitude (real physics
    feature -- only the normalisation is a numerical convenience)."""
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / fs).to(x.device)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    band_power = (spec.abs() ** 2)[..., mask].mean(dim=-1)
    total_power = (spec.abs() ** 2).mean(dim=-1).clamp_min(1e-8)
    return (band_power / total_power).mean(dim=1)  # average over channels -> (batch,)


def main():
    train, test, labels, mean, std, k = make_dataset()
    train, test = train.to(DEVICE), test.to(DEVICE)
    fr, bpfo, bpfi, bsf = bearing_fault_frequencies()
    print(f"Real bearing kinematics at {RPM:.0f} RPM: shaft={fr:.2f} Hz, "
          f"BPFO={bpfo:.2f} Hz, BPFI={bpfi:.2f} Hz, BSF={bsf:.2f} Hz")

    model = MaskedAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    history = []

    shaft_lo, shaft_hi = fr * 0.8, fr * 1.2  # physics band: rotational-order consistency

    for epoch in range(1, 81):
        model.train(); opt.zero_grad()
        reconstruction = model(random_mask(train))
        reconstruction_loss = F.mse_loss(reconstruction, train)
        physics_loss = F.mse_loss(
            band_energy(reconstruction, FS, shaft_lo, shaft_hi),
            band_energy(train, FS, shaft_lo, shaft_hi),
        )
        loss = reconstruction_loss + 2.0 * physics_loss
        loss.backward(); opt.step(); history.append(loss.item())
        if epoch in {1, 20, 40, 60, 80}:
            print(f"epoch={epoch:03d} total={loss.item():.5f} reconstruction={reconstruction_loss.item():.5f} physics={physics_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        train_rec = model(train)
        test_rec = model(test)
        # Real physics-informed anomaly evidence: energy at real fault bands.
        fault_band_energy = (
            band_energy(test, FS, bpfo - 5, bpfo + 5)
            + band_energy(test, FS, bpfi - 5, bpfi + 5)
            + band_energy(test, FS, bsf - 5, bsf + 5)
        )
        train_fault_band = (
            band_energy(train, FS, bpfo - 5, bpfo + 5)
            + band_energy(train, FS, bpfi - 5, bpfi + 5)
            + band_energy(train, FS, bsf - 5, bsf + 5)
        )
        fb_scale = train_fault_band.mean().clamp_min(1e-8)

        train_scores = (train_rec - train).pow(2).mean((1, 2)) + 0.02 * (train_fault_band / fb_scale)
        test_scores = (test_rec - test).pow(2).mean((1, 2)) + 0.02 * (fault_band_energy / fb_scale)

        threshold = torch.quantile(train_scores, 0.95)
        pred = (test_scores > threshold).float()
        true_binary = (labels != 0).float()
        accuracy = (pred.cpu() == true_binary).float().mean().item()
        healthy_fp = pred[:k].mean().item() * 100
        fault_recall = pred[k:].mean().item() * 100
    print(f"threshold={threshold.item():.4f}")
    print(f"test_accuracy={accuracy:.3f}")
    print(f"healthy_false_alarm_rate={healthy_fp:.1f}%")
    print(f"fault_recall={fault_recall:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = ["tab:blue"] * k + ["tab:orange"] * k + ["tab:green"] * k + ["tab:red"] * k
    axes[0].scatter(range(len(test_scores)), test_scores.cpu(), c=colors, s=16)
    axes[0].axhline(threshold.cpu(), linestyle="--", color="black", label="95% healthy threshold")
    for b in (k, 2 * k, 3 * k):
        axes[0].axvline(b - 0.5, linestyle=":", color="gray")
    axes[0].set(xlabel="test window (Normal | Inner race | Outer race | Ball)",
                ylabel="anomaly score", title="Real CWRU bearing anomaly scores")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    idx = k + 3  # an inner-race fault window
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    spec_fault = np.abs(np.fft.rfft(test[idx, 0].cpu().numpy()))
    spec_normal = np.abs(np.fft.rfft(test[0, 0].cpu().numpy()))
    axes[1].plot(freqs, spec_normal, label="healthy DE spectrum", alpha=0.8)
    axes[1].plot(freqs, spec_fault, label="inner-race fault DE spectrum", alpha=0.8)
    for name, f0 in [("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf)]:
        axes[1].axvline(f0, color="red", linestyle=":", alpha=0.6)
        axes[1].text(f0, axes[1].get_ylim()[1] * 0.9, name, rotation=90, fontsize=7)
    axes[1].set(xlim=(0, 500), xlabel="frequency [Hz]", ylabel="|FFT|",
                title="Real bearing fault frequencies vs. measured spectrum")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "plots/task3_anomaly.png", dpi=180); plt.close(fig)

    (OUT / "outputs/task3.txt").write_text(
        "Dataset: CWRU Bearing Data Center, 1797 RPM, DE+FE channels (real)\n"
        f"Real bearing frequencies: shaft={fr:.2f} Hz, BPFO={bpfo:.2f} Hz, "
        f"BPFI={bpfi:.2f} Hz, BSF={bsf:.2f} Hz\n"
        f"Threshold: {threshold.item():.4f}\nAccuracy: {accuracy:.3f}\n"
        f"Healthy false-alarm rate: {healthy_fp:.1f}%\nFault recall: {fault_recall:.1f}%\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
