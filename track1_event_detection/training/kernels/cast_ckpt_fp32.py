"""Cast a P2-silver `ckpt-<n>.pt` (fp16 weights) to float32 in the rebuild format.

    python cast_ckpt_fp32.py <ckpt-n.pt> <out.pt> [--compare <fp32.pt>]

The kernel keeps every evaluated checkpoint as fp16 weights only
(payload['model_state'], floating tensors .half()); inference/export_frames.py reads
payload['model_state'] into P2's architecture. This script writes the same layout with every floating tensor cast to
float32 (non-floating tensors unchanged) and the source metadata kept. `--compare`
reports the largest absolute weight difference against an fp32 checkpoint of the same
update (e.g. best.pt), i.e. the fp16 rounding that the cast cannot undo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('out')
    parser.add_argument('--compare', default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    state = payload['model_state']
    dtypes = sorted({str(v.dtype) for v in state.values()})
    cast = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    result = {key: payload.get(key) for key in ('arm', 'updates', 'metrics', 'config', 'decoders',
                                                'architecture')}
    result.update({'kind': 'indoml_p2_silver_ckpt_fp32_cast_v1', 'dtype': 'float32',
                   'cast_from': dtypes, 'source': str(source), 'source_sha256': sha256(source),
                   'source_kind': payload.get('kind'), 'model_state': cast})
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + '.tmp')
    torch.save(result, temporary)
    temporary.replace(out)
    report = {'source': str(source), 'out': str(out), 'updates': payload.get('updates'),
              'tensors': len(cast), 'source_dtypes': dtypes,
              'parameters': int(sum(v.numel() for v in cast.values() if v.is_floating_point())),
              'metrics': payload.get('metrics'), 'out_sha256': sha256(out)}
    if args.compare:
        reference = torch.load(args.compare, map_location='cpu', weights_only=False)
        reference = reference.get('model_state', reference)
        assert set(reference) == set(cast), 'Key sets differ'
        worst, worst_key, relative = 0.0, None, 0.0
        for key, value in cast.items():
            if not value.is_floating_point():
                assert torch.equal(value, reference[key]), key
                continue
            diff = (value - reference[key].float()).abs().max().item()
            if diff > worst:
                worst, worst_key = diff, key
                scale = reference[key].float().abs().max().item()
                relative = diff / scale if scale else 0.0
        report['compare'] = {'against': args.compare, 'max_abs_diff': worst,
                             'max_abs_diff_tensor': worst_key,
                             'max_abs_diff_relative_to_tensor_max': relative}
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
