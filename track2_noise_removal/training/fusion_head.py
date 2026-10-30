"""Learned fusion head: a small conv net that sets the ZipEnhancer weight w(t, f) per TF bin of the fused output:
    Y = w Z + (1 - w) G,   Z = STFT(z), G = STFT(g t), g = <z, t> / <t, t> (the deployed LS gain, detached),
STFT n_fft 512, hop 128, Hann, centre; features per bin: log|X|, log|Z|, log|G| (each minus the clip mean of log|X|)
and cos(angle Z - angle G); 4 conv2d 3x3 layers of 32 channels + ReLU, then a 1x1 conv; w = sigmoid(out). The last layer
starts at weight 0 and bias logit(0.70): an untrained head gives exactly the plain average 0.70 z + 0.30 g t (printed
as 'E1 average' below).
Training data: the shards of training/dump_fusion_mixtures.py (fresh training mixtures, x / s / z / t at the training
scale); only crops whose ZipEnhancer pt20 pass-through share is < 0.20 (the fusion route of the route rule); loss
-SI-SNR(istft(Y), s); 10 % of the crops are a held-out check (training distribution only: no validation clip).
The head of the last epoch is used.

usage:
  fusion_head.py train DUMP_DIR OUT_PT [EPOCHS 20] [LR 1e-3]     -> OUT_PT (state_dict + meta), OUT_PT.log.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))   # pp_lib.py is next to this file
import pp_lib  # noqa: E402

NFFT, HOP = 512, 128
W0 = 0.70


class Head(nn.Module):
    def __init__(self, ch=32, w0=W0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(w0 / (1 - w0)))

    def weight(self, X, Z, G):
        eps = 1e-5
        lx, lz, lg = (torch.log(a.abs() + eps) for a in (X, Z, G))
        mu = lx.mean(dim=(1, 2), keepdim=True)
        cd = torch.cos(torch.angle(Z) - torch.angle(G))
        f = torch.stack([lx - mu, lz - mu, lg - mu, cd], 1)
        return torch.sigmoid(self.net(f))[:, 0]

    def forward(self, X, Z, G):
        w = self.weight(X, Z, G)
        return w * Z + (1 - w) * G


def stft(a):
    return torch.stft(a, NFFT, HOP, window=torch.hann_window(NFFT, device=a.device), center=True, return_complex=True)


def istft(A, n):
    return torch.istft(A, NFFT, HOP, window=torch.hann_window(NFFT, device=A.device), center=True, length=n)


def fuse(head, x, z, t):
    """x, z, t: (B, n) float tensors -> (B, n) fused output (g per row as the deployment: <z, t> / <t, t>)."""
    g = (z * t).sum(-1, keepdim=True) / ((t * t).sum(-1, keepdim=True) + 1e-9)
    Y = head(stft(x), stft(z), stft(g * t))
    return istft(Y, x.shape[-1])


def si_snr_loss(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) * ref / ((ref ** 2).sum(-1, keepdim=True) + eps)
    e = est - s
    return -(10 * torch.log10((s ** 2).sum(-1) / ((e ** 2).sum(-1) + eps) + eps)).mean()


def load_dump(d):
    """Shard by shard: keep only the crops on the fusion route, in float16 (keeps the memory low; float16 -> float32
    per batch is exact)."""
    parts = {'x': [], 's': [], 'z': [], 't': []}
    n_all = 0
    for p in sorted(Path(d).glob('shard_*.npz')):
        f = np.load(p)
        x, z = f['x'], f['z']
        keep = np.array([pp_lib.pp(z[i].astype(np.float64) * 32768, x[i].astype(np.float64) * 32768, gate=False,
                                   T=20)[2] < 0.20 for i in range(len(x))])   # the AVG route: Zip pt20 share < 0.20
        n_all += len(x)
        for k in parts:
            parts[k].append(f[k][keep])
        del f, x, z
    out = []
    for k in ('x', 's', 'z', 't'):
        out.append(np.concatenate(parts[k]))
        parts[k] = None
    return out[0], out[1], out[2], out[3], len(out[0]), n_all


def train(dump, out_pt, epochs=20, lr=1e-3):
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(1234)
    X, S, Z, T, n_keep, n_all = load_dump(dump)
    rng = np.random.default_rng(1234)
    idx = rng.permutation(n_keep)
    n_va = max(1, n_keep // 10)
    va, tr = idx[:n_va], idx[n_va:]
    head = Head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    log = {'n_all': n_all, 'n_avg_route': n_keep, 'n_train': len(tr), 'n_check': len(va), 'epochs': []}

    def batch(ii):
        return (torch.from_numpy(a[ii].astype(np.float32)).to(dev) for a in (X, S, Z, T))   # float16 -> float32 is exact

    def check():
        head.eval()
        with torch.no_grad():
            ls, l0 = [], []
            for j in range(0, len(va), 16):
                x, s, z, t = batch(va[j:j + 16])
                ls.append(float(si_snr_loss(fuse(head, x, z, t), s)) * len(x))
                g = (z * t).sum(-1, keepdim=True) / ((t * t).sum(-1, keepdim=True) + 1e-9)
                l0.append(float(si_snr_loss(W0 * z + (1 - W0) * g * t, s)) * len(x))
        head.train()
        return sum(ls) / len(va), sum(l0) / len(va)

    c0 = check()
    log['check_start'] = {'head': c0[0], 'e1_average': c0[1]}
    print(f'{n_all} mixtures, {n_keep} on the AVG route ({len(tr)} train / {len(va)} check); check -SI-SNR head '
          f'{c0[0]:.4f} vs E1 average {c0[1]:.4f}', flush=True)
    t0 = time.time()
    for ep in range(epochs):
        p = rng.permutation(tr)
        tot = 0.0
        for j in range(0, len(p), 16):
            x, s, z, t = batch(p[j:j + 16])
            loss = si_snr_loss(fuse(head, x, z, t), s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(x)
        c = check()
        log['epochs'].append({'epoch': ep + 1, 'train': tot / len(tr), 'check_head': c[0], 'check_e1': c[1],
                              'sec': time.time() - t0})
        print(f'epoch {ep + 1}: train {tot / len(tr):.4f}, check head {c[0]:.4f} (E1 average {c[1]:.4f}), '
              f'{time.time() - t0:.0f} s', flush=True)
    torch.save({'state_dict': head.state_dict(), 'w0': W0, 'nfft': NFFT, 'hop': HOP}, out_pt)
    Path(str(out_pt) + '.log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20,
              float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3)
    else:
        raise SystemExit(__doc__)
