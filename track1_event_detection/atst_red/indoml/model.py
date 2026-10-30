"""Small binary CRNN, with CPU log-mel extraction and explicit frame alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from .data import load_audio, read_manifest, resolve_audio_path


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 320
    n_mels: int = 64

    @property
    def frame_hop(self) -> float:
        return self.hop_length / self.sample_rate


class LogMelFrontend(nn.Module):
    """HTK triangular mel power; fixed log scaling, no per-clip peak normalization.

    STFT frame i is centered at i * hop / sample_rate. Constant padding gives
    floor(samples / hop) + 1 frames, including for clips shorter than one window.
    Run this frontend on CPU; only real-valued features move to MPS/CUDA.
    """

    def __init__(self, config: FrontendConfig | None = None):
        super().__init__()
        self.config = config or FrontendConfig()
        c = self.config
        if not (0 < c.hop_length <= c.win_length <= c.n_fft and c.n_mels > 0):
            raise ValueError("Invalid STFT or mel configuration")
        mel_max = 2595 * np.log10(1 + (c.sample_rate / 2) / 700)
        hz = 700 * (10 ** (torch.linspace(0, mel_max, c.n_mels + 2) / 2595) - 1)
        frequencies = torch.linspace(0, c.sample_rate / 2, c.n_fft // 2 + 1)
        left = (frequencies[None, :] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
        right = (hz[2:, None] - frequencies[None, :]) / (hz[2:] - hz[1:-1])[:, None]
        filters = torch.minimum(left, right).clamp_min(0)
        filters = filters / filters.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("filters", filters)
        self.register_buffer("window", torch.hann_window(c.win_length))

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 1 or not audio.numel() or not torch.isfinite(audio).all():
            raise ValueError("Expected nonempty, finite mono audio")
        c = self.config
        spectrum = torch.stft(audio.float(), n_fft=c.n_fft, hop_length=c.hop_length,
                              win_length=c.win_length, window=self.window,
                              center=True, pad_mode="constant", return_complex=True)
        mel = self.filters @ spectrum.abs().square()
        features = ((mel.clamp_min(1e-8).log10() + 4) / 4).transpose(0, 1)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite log-mel features")
        return features.contiguous()


class NoiseCRNN(nn.Module):
    """Frequency-only pooling preserves the 20 ms timeline.

    Masking after each convolution prevents padding from leaking into valid
    frames. Packed recurrence sees only valid frames in each clip.
    """

    def __init__(self, n_mels: int = 64, channels: tuple = (16, 24, 32), hidden: int = 64):
        super().__init__()
        if n_mels < 2 ** len(channels):
            raise ValueError("Too few mel bins for frequency pooling")
        self.config = {"n_mels": n_mels, "channels": list(channels), "hidden": hidden}
        sizes = [1, *channels]
        self.convolutions = nn.ModuleList([
            nn.Conv2d(sizes[i], sizes[i + 1], kernel_size=3, padding=1)
            for i in range(len(channels))
        ])
        self.gru = nn.GRU(channels[-1] * (n_mels // 2 ** len(channels)), hidden,
                          batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, 1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or lengths.numel() != features.shape[0]:
            raise ValueError("Expected features [batch, frames, mel] and one length per clip")
        valid_lengths = [int(n) for n in lengths.detach().cpu().tolist()]
        if any(n <= 0 or n > features.shape[1] for n in valid_lengths):
            raise ValueError("Invalid feature lengths")
        mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths.to(features.device)[:, None]
        x = features.unsqueeze(1) * mask[:, None, :, None]
        for convolution in self.convolutions:
            x = F.max_pool2d(F.relu(convolution(x)), kernel_size=(1, 2))
            x = x * mask[:, None, :, None]
        x = x.permute(0, 2, 1, 3).flatten(2)
        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        return self.classifier(encoded)[:, :, 0] * mask


QUALITY_WEIGHTS = {"verified_timestamps": 1.0, "unverified_timestamps": 0.3,
                   "preview_timestamps": 1.0}


def frame_targets(events: list[dict], frames: int, frame_hop: float) -> torch.Tensor:
    centers = torch.arange(frames, dtype=torch.float64) * frame_hop
    targets = torch.zeros(frames)
    for event in events:
        targets[(centers >= float(event["onset"])) & (centers < float(event["offset"]))] = 1
    return targets


class ClipDataset(Dataset):
    def __init__(self, manifest: str | Path, config: FrontendConfig, require_labels: bool = True):
        self.manifest = Path(manifest).resolve()
        rows = read_manifest(self.manifest)
        self.skipped = 0
        self.rows = []
        for row in rows:
            if require_labels and (row.get("events") is None or row.get("quality") == "no_timestamps"):
                self.skipped += 1
                continue
            if require_labels and row.get("quality") not in QUALITY_WEIGHTS:
                raise ValueError(f"Unsupported labeled quality: {row.get('quality')!r}")
            self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No usable clips in {self.manifest}; skipped {self.skipped} without timestamps")
        self.config = config
        self.frontend = LogMelFrontend(config)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = resolve_audio_path(row, self.manifest)
        waveform = load_audio(path, sample_rate=self.config.sample_rate)
        duration = len(waveform) / self.config.sample_rate
        if abs(duration - float(row["duration"])) > 0.05:
            raise ValueError(f"Duration disagrees with audio for {row['clip_id']}: manifest={row['duration']}, audio={duration}")
        events = row.get("events")
        if events and any(float(e["offset"]) > duration + 0.02 for e in events):
            raise ValueError(f"Event extends beyond audio for {row['clip_id']}")
        features = self.frontend(torch.from_numpy(np.asarray(waveform, dtype=np.float32)))
        return {"features": features, "targets": frame_targets(events or [], len(features), self.config.frame_hop),
                "weight": QUALITY_WEIGHTS.get(row.get("quality"), 1.0),
                "clip_id": row["clip_id"], "duration": duration, "events": events}

    def positive_weight(self, cap: float = 20.0) -> float:
        """Estimate from training annotation frames only, weighted by label tier."""
        positive = total = 0.0
        for row in self.rows:
            frames = round(float(row["duration"]) * self.config.sample_rate) // self.config.hop_length + 1
            weight = QUALITY_WEIGHTS[row["quality"]]
            positive += frame_targets(row["events"], frames, self.config.frame_hop).sum().item() * weight
            total += frames * weight
        return min(cap, max(1.0, (total - positive) / max(positive, 1.0)))


def collate_clips(clips: list[dict]) -> dict:
    lengths = torch.tensor([len(c["features"]) for c in clips], dtype=torch.long)
    features = nn.utils.rnn.pad_sequence([c["features"] for c in clips], batch_first=True)
    targets = nn.utils.rnn.pad_sequence([c["targets"] for c in clips], batch_first=True)
    mask = torch.arange(features.shape[1])[None, :] < lengths[:, None]
    return {"features": features, "targets": targets, "lengths": lengths, "mask": mask,
            "weights": torch.tensor([c["weight"] for c in clips]), "clips": clips}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = FrontendConfig(**checkpoint["frontend"])
    model = NoiseCRNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), config, checkpoint
