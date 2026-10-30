"""Per-frame outputs of one member checkpoint (S1, R10, PSR10B and their later rounds) at one input shift.

Every member has exactly P2's architecture, so P2 is built with its own loader (context_crop.load_parent) and
its weights are replaced by the member checkpoint's 'model_state'. Everything after that (packing,
api.infer_outputs, the npz layout, the input shift) is indoml_pipeline.p2_infer, unchanged: only `load_p2`
is rebound in this process. CPU, fp32.

  python export_frames.py --checkpoint member.pt --audio DIR --out frames_shift160.npz \
      [--shift 160] [--clips list.txt] [--threads 4] [--report r.json]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indoml_pipeline import p2_infer  # noqa: E402


def make_loader(checkpoint):
    checkpoint = Path(checkpoint)

    def load_rebuilt(device='cpu'):
        p2_infer._prepare_sys_path()
        import context_crop
        import torch
        api = p2_infer.api_module()
        model, metadata = context_crop.load_parent(api, p2_infer.NATIVE, p2_infer.CHECKPOINT, device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload.get('model_state', payload)
        missing, unexpected = model.load_state_dict(state, strict=True)
        assert not missing and not unexpected
        model.eval()
        model.frequency_augmentation = None
        metadata = dict(metadata or {})
        metadata['rebuilt_checkpoint'] = str(checkpoint)
        metadata['rebuilt_state_digest'] = api.state_digest(model.state_dict())
        metadata['rebuilt_payload_keys'] = sorted(k for k in payload if k != 'model_state')
        return model, metadata

    return load_rebuilt


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--checkpoint' not in argv:
        raise SystemExit('--checkpoint is required')
    i = argv.index('--checkpoint')
    checkpoint = argv[i + 1]
    del argv[i:i + 2]
    p2_infer.load_p2 = make_loader(checkpoint)
    with Path(checkpoint).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps({'rebuilt_checkpoint': checkpoint, 'sha256': digest}), flush=True)
    p2_infer.main(argv)


if __name__ == '__main__':
    main()
