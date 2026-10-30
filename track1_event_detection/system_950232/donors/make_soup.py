"""make_soup.py: uniform fp32 weight average ("model soup") of two P2-architecture checkpoints (the SOUP2 rule).

    python make_soup.py --a A.pt --b B.pt --out SOUP.pt --report SOUP.json
        [--a-sha256 HEX] [--b-export-report EXPORT.json] [--require-b-export-report] [--b-rounding match-a|none]

Inputs. Each file is loaded on the CPU (torch.load map_location='cpu', mmap). The model state is the one the stack
loaders use: payload['model_state'] when present, else the payload itself when it is a bare {name: tensor} dict
(inference/export_frames.py and psr10_port.load_weights read it the same way). Floating tensors
are cast to float32, as both loaders do (load_state_dict into P2's fp32 model; load_weights .float()).
    Kaggle fp32 cast file (training/kernels/cast_ckpt_fp32.py, kind indoml_p2_silver_ckpt_fp32_cast_v1): model_state
        float32 whose values are fp16 values (cast from the kernel's fp16 ckpt-<u>.pt).
    Mac state file (mac_train.py final-<T>.pt / plateau-<D>.pt / last.pt, kind mac_psr10_state_v1): model_state
        float32 master weights (+ optimizer etc., not read). The Mac T export uses final-<T>.pt (export-report.json).
    Mac checkpoint (mac_train.py ckpt-<u>.pt, kind mac_psr10_checkpoint_v1): model_state float16 = .half() of the
        state at update u, with state_digest_fp32 = the digest of that fp32 state.

b = the weights that b's export uses. The export report (--b-export-report, or found at
<dir of b>/export/u<updates>/export-report.json) must name b: its checkpoint.source is the --b file, its sha256
equals the file's sha256 and its state_digest equals the digest of b's fp32 state (else STOP).

fp16 rounding (the fixed rule "with the same fp16 rounding as (a)", --b-rounding match-a, the default): when every
floating value of a is an fp16 value (a is an fp16 checkpoint cast to fp32) and b is fp32 only (not fp16-rounded), b
is replaced by fp32(fp16(b)). It is taken through b's own fp16 checkpoint path when <dir of b>/ckpt-<updates>.pt
exists: that file must equal fp16(b) tensor for tensor and carry state_digest_fp32 = digest(b) (else STOP); without
that file b is rounded with .half(), which is the same operation. When a is not fp16-rounded, or b already is, b is
used unchanged. --b-rounding none uses b exactly as exported (recorded in the report as a deviation).

Checks (STOP, exit 2, no output file): key sets, shapes and dtypes (of the fp32 states the loaders build) equal;
the raw file dtypes agree per tensor up to the fp16 -> fp32 cast; every non-floating tensor (integer/bool buffer)
equal; every floating value finite (inputs, rounded b, soup); --a-sha256 matches when given.

Soup: soup = (a + b) * 0.5 in float32 for every floating tensor (one rounding of the sum; the halving is exact);
non-floating tensors are copied (they are equal). The output is the format that
inference/export_frames.py loads (payload['model_state'], strict): {'kind': 'soup_r10_uniform_fp32_v1', 'dtype': 'float32',
'model_state': ..., + provenance}. It is written to <out>.tmp, renamed, loaded again and compared with the
in-memory soup (torch.equal per tensor). An existing --out or --report is never overwritten (exit 2).

Report (information only): sha256 of the inputs and the output, the digests,
the tensor and parameter counts, the rounding decision, per group (frontend, encoder embeddings, each encoder block,
encoder norm, head) the max and mean |a - b| (b after the rounding rule), the relative L2 distance, and the size of
the fp16 rounding of b; the timings and the peak RSS.

CPU only (CUDA hidden, no MPS), 3 threads, no byte-code files. Writes only --out (and its .tmp) and --report.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = os.environ.get('SOUP_THREADS', '3')

import argparse  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

TOOL_VERSION = 'make_soup.py v1 (2026-09-24)'
STARTED = time.time()


class Stop(Exception):
    """A required check failed: no soup."""


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    """= indoml.atst_intervals.state_digest (names, dtype, shape, bytes; sorted by name)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)     # bytes on macOS


def load_payload(torch, path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True), True
    except RuntimeError:
        return torch.load(path, map_location='cpu', weights_only=False), False


def find_state(torch, payload, label):
    if isinstance(payload, dict) and isinstance(payload.get('model_state'), dict):
        state, where = payload['model_state'], "payload['model_state']"
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        state, where = payload, 'bare state dict'
    else:
        keys = sorted(payload)[:20] if isinstance(payload, dict) else type(payload).__name__
        raise Stop(f'{label}: no model state (neither payload["model_state"] nor a bare state dict); keys {keys}')
    bad = [k for k, v in state.items() if not torch.is_tensor(v)]
    if bad:
        raise Stop(f'{label}: model state has non-tensor entries {bad[:5]}')
    return state, where


def meta_of(payload):
    if not isinstance(payload, dict) or 'model_state' not in payload:
        return {}
    keep = ('kind', 'run', 'arm', 'updates', 'dtype', 'source', 'source_sha256', 'source_kind', 'cast_from',
            'state_digest_fp32', 'amp', 'T', 'D', 'utc', 'architecture')
    out = {}
    for key in keep:
        if key in payload:
            value = payload[key]
            out[key] = value if isinstance(value, (str, int, float, bool, type(None), list)) else repr(value)[:200]
    return out


def fp32_state(torch, state):
    return {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}


def fp16_rounded(torch, state):
    """True when every floating tensor of the fp32 state holds only fp16 values."""
    return all(torch.equal(v, v.half().float()) for v in state.values() if v.is_floating_point())


def group_of(name):
    parts = name.split('.')
    if parts[0] == 'encoder':
        if len(parts) > 2 and parts[1] == 'blocks':
            return f'encoder.blocks.{int(parts[2]):02d}'
        if parts[1] in ('mask_embed', 'pos_embed', 'patch_embed'):
            return 'encoder.embed'
        return f'encoder.{parts[1]}'
    return parts[0]


def compare_structure(torch, raw_a, raw_b, eff_a, eff_b):
    """The required checks; returns the check record or raises Stop."""
    keys_a, keys_b = set(eff_a), set(eff_b)
    if keys_a != keys_b:
        raise Stop(f'key sets differ: only in a {sorted(keys_a - keys_b)[:10]}, only in b {sorted(keys_b - keys_a)[:10]}'
                   f' ({len(keys_a - keys_b)} / {len(keys_b - keys_a)})')
    shapes = [k for k in eff_a if tuple(eff_a[k].shape) != tuple(eff_b[k].shape)]
    if shapes:
        raise Stop('shapes differ: ' + '; '.join(f'{k} {tuple(eff_a[k].shape)} vs {tuple(eff_b[k].shape)}'
                                                 for k in shapes[:10]))
    dtypes = [k for k in eff_a if eff_a[k].dtype != eff_b[k].dtype]
    if dtypes:
        raise Stop('dtypes differ (fp32 loader states): ' + '; '.join(f'{k} {eff_a[k].dtype} vs {eff_b[k].dtype}'
                                                                      for k in dtypes[:10]))
    raw_mismatch = []
    for k in eff_a:
        ra, rb = raw_a[k].dtype, raw_b[k].dtype
        if ra == rb:
            continue
        if raw_a[k].is_floating_point() and raw_b[k].is_floating_point() and {ra, rb} <= {torch.float16, torch.float32}:
            continue                                    # the fp16 -> fp32 cast of the fp16 checkpoint path
        raw_mismatch.append(f'{k} {ra} vs {rb}')
    if raw_mismatch:
        raise Stop('raw dtypes differ beyond the fp16 -> fp32 cast: ' + '; '.join(raw_mismatch[:10]))
    integer = sorted(k for k, v in eff_a.items() if not v.is_floating_point())
    unequal = [k for k in integer if not torch.equal(eff_a[k], eff_b[k])]
    if unequal:
        raise Stop(f'integer/bool buffers differ: {unequal[:10]}')
    for label, state in (('a', eff_a), ('b', eff_b)):
        bad = [k for k, v in state.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
        if bad:
            raise Stop(f'{label}: non-finite values in {bad[:10]}')
    return {'key_sets_equal': True, 'shapes_equal': True, 'dtypes_equal_fp32_states': True,
            'raw_dtypes': {'a': sorted({str(v.dtype) for v in raw_a.values()}),
                           'b': sorted({str(v.dtype) for v in raw_b.values()})},
            'raw_dtypes_agree_up_to_fp16_cast': True,
            'integer_buffers': len(integer), 'integer_buffers_equal': True, 'inputs_finite': True,
            'tensors': len(eff_a)}


def resolve_export_report(args, path_b, payload_b):
    if args.b_export_report:
        return Path(args.b_export_report)
    updates = payload_b.get('updates') if isinstance(payload_b, dict) else None
    if isinstance(updates, int):
        candidate = path_b.parent / 'export' / f'u{updates:06d}' / 'export-report.json'
        if candidate.is_file():
            return candidate
    return None


def check_export_link(report_path, path_b, sha_b, digest_b):
    report = json.loads(Path(report_path).read_text())
    checkpoint = report.get('checkpoint') or {}
    source = checkpoint.get('source')
    record = {'export_report': str(report_path), 'export_report_sha256': sha256_file(report_path),
              'export_checkpoint': checkpoint, 'export_update': report.get('update'),
              'export_amp': report.get('amp'), 'export_splits': report.get('splits'),
              'export_shifts': report.get('shifts')}
    same_file = bool(source) and Path(source).exists() and os.path.samefile(source, path_b)
    record['source_is_b'] = same_file
    record['sha256_matches'] = checkpoint.get('sha256') == sha_b
    record['state_digest_matches'] = checkpoint.get('state_digest') == digest_b
    if not same_file:
        raise Stop(f'b is not the file its export used: export-report checkpoint.source = {source}, --b = {path_b}')
    if not record['sha256_matches']:
        raise Stop(f'b sha256 {sha_b} != export-report sha256 {checkpoint.get("sha256")}')
    if not record['state_digest_matches']:
        raise Stop(f'b state digest {digest_b} != export-report state_digest {checkpoint.get("state_digest")}')
    return record


def apply_rounding(torch, args, eff_a, eff_b, path_b, payload_b, digest_b):
    """The fixed fp16 rounding rule; returns (b used in the soup, decision record)."""
    a_rounded, b_rounded = fp16_rounded(torch, eff_a), fp16_rounded(torch, eff_b)
    decision = {'rule': args.b_rounding, 'a_fp16_rounded': a_rounded, 'b_fp16_rounded_as_given': b_rounded}
    if args.b_rounding == 'none':
        decision['action'] = 'b used exactly as its export used it (--b-rounding none: NOT the default rule)'
        return eff_b, decision
    if not a_rounded or b_rounded:
        decision['action'] = ('b used unchanged: ' + ('a is not fp16-rounded' if not a_rounded
                                                       else 'b is already fp16-rounded (same rounding as a)'))
        return eff_b, decision
    rounded = {k: (v.half().float() if v.is_floating_point() else v) for k, v in eff_b.items()}
    bad = [k for k, v in rounded.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise Stop(f'b: fp16 rounding overflows in {bad[:10]}')
    updates = payload_b.get('updates') if isinstance(payload_b, dict) else None
    ckpt = path_b.parent / f'ckpt-{updates:06d}.pt' if isinstance(updates, int) else None
    decision['b_fp16_checkpoint'] = str(ckpt) if ckpt else None
    if ckpt is not None and ckpt.is_file() and not os.path.samefile(ckpt, path_b):
        payload16, _ = load_payload(torch, ckpt)
        state16, _ = find_state(torch, payload16, 'b fp16 checkpoint')
        if set(state16) != set(eff_b):
            raise Stop(f'{ckpt}: key set differs from b')
        differ = [k for k, v in eff_b.items()
                  if not (state16[k].dtype == torch.float16 and torch.equal(state16[k], v.half()))
                  and v.is_floating_point()]
        differ += [k for k, v in eff_b.items() if not v.is_floating_point() and not torch.equal(state16[k], v)]
        recorded = payload16.get('state_digest_fp32') if isinstance(payload16, dict) else None
        decision.update({'b_fp16_checkpoint_sha256': sha256_file(ckpt),
                         'b_fp16_checkpoint_kind': payload16.get('kind') if isinstance(payload16, dict) else None,
                         'b_fp16_checkpoint_updates': payload16.get('updates') if isinstance(payload16, dict) else None,
                         'b_fp16_checkpoint_state_digest_fp32': recorded,
                         'b_fp16_checkpoint_equals_half_of_b': not differ,
                         'b_fp16_checkpoint_digest_matches_b': recorded == digest_b})
        if differ:
            raise Stop(f'{ckpt} is not the fp16 image of b: {len(differ)} tensors differ, e.g. {differ[:5]}')
        if recorded is not None and recorded != digest_b:
            raise Stop(f'{ckpt} state_digest_fp32 {recorded} != digest of b {digest_b}')
        rounded = {k: (v.float() if v.is_floating_point() else v) for k, v in state16.items()}
        decision['action'] = ('b is fp32 only and a is fp16-rounded: b = fp32 cast of its own fp16 checkpoint '
                              f'{ckpt.name} (verified = fp16(b), same rounding as a)')
        del payload16, state16
    else:
        decision['action'] = 'b is fp32 only and a is fp16-rounded: b = fp32(fp16(b)) with .half() (no fp16 checkpoint file)'
    diffs = [float((rounded[k] - v).abs().max()) for k, v in eff_b.items() if v.is_floating_point() and v.numel()]
    decision['max_abs_fp16_rounding_of_b'] = max(diffs) if diffs else 0.0
    decision['max_abs_soup_change_vs_rule_none'] = 0.5 * decision['max_abs_fp16_rounding_of_b']
    return rounded, decision


def group_stats(torch, eff_a, used_b, eff_b):
    groups = {}
    for key, va in eff_a.items():
        if not va.is_floating_point():
            continue
        vb = used_b[key]
        diff = (va - vb).abs()
        g = groups.setdefault(group_of(key), {'tensors': 0, 'params': 0, 'max_abs_a_minus_b': 0.0, 'sum_abs': 0.0,
                                               'sum_sq_diff': 0.0, 'sum_sq_a': 0.0, 'tensors_equal': 0,
                                               'max_abs_b_rounding': 0.0})
        g['tensors'] += 1
        g['params'] += va.numel()
        if va.numel():
            g['max_abs_a_minus_b'] = max(g['max_abs_a_minus_b'], float(diff.max()))
            g['sum_abs'] += float(diff.double().sum())
            g['sum_sq_diff'] += float(diff.double().pow(2).sum())
            g['sum_sq_a'] += float(va.double().pow(2).sum())
            g['max_abs_b_rounding'] = max(g['max_abs_b_rounding'], float((vb - eff_b[key]).abs().max()))
        g['tensors_equal'] += int(torch.equal(va, vb))
    out, total = {}, {'tensors': 0, 'params': 0, 'sum_abs': 0.0, 'max_abs_a_minus_b': 0.0,
                      'sum_sq_diff': 0.0, 'sum_sq_a': 0.0}
    for name in sorted(groups):
        g = groups[name]
        out[name] = {'tensors': g['tensors'], 'params': g['params'],
                     'max_abs_a_minus_b': g['max_abs_a_minus_b'],
                     'mean_abs_a_minus_b': g['sum_abs'] / g['params'] if g['params'] else 0.0,
                     'rel_l2_a_minus_b': (g['sum_sq_diff'] ** .5) / (g['sum_sq_a'] ** .5) if g['sum_sq_a'] else None,
                     'tensors_equal_a_b': g['tensors_equal'],
                     'max_abs_fp16_rounding_of_b': g['max_abs_b_rounding']}
        for key in ('tensors', 'params', 'sum_abs', 'sum_sq_diff', 'sum_sq_a'):
            total[key] += g[key]
        total['max_abs_a_minus_b'] = max(total['max_abs_a_minus_b'], g['max_abs_a_minus_b'])
    overall = {'tensors': total['tensors'], 'params': total['params'], 'max_abs_a_minus_b': total['max_abs_a_minus_b'],
               'mean_abs_a_minus_b': total['sum_abs'] / total['params'] if total['params'] else 0.0,
               'rel_l2_a_minus_b': (total['sum_sq_diff'] ** .5) / (total['sum_sq_a'] ** .5) if total['sum_sq_a'] else None}
    return out, overall


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='member a (R10 at update 15,500 in fp32, e.g. R10/ckpt-015500-fp32.pt)')
    p.add_argument('--b', required=True, help="member b = the file b's T export used (export-report.json source)")
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--a-sha256', default=None, help='the expected sha256 of a (STOP when it differs)')
    p.add_argument('--b-export-report', default=None, help='default: <dir of b>/export/u<updates>/export-report.json')
    p.add_argument('--require-b-export-report', action='store_true')
    p.add_argument('--b-rounding', choices=('match-a', 'none'), default='match-a')
    p.add_argument('--label', default=None, help='free text stored in the report and the payload (e.g. PIPELINE TEST)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    out, report_path = Path(args.out), Path(args.report)
    for target in (out, report_path, out.with_name(out.name + '.tmp')):
        if target.exists():
            print(json.dumps({'status': 'REFUSED', 'error': f'{target} exists (outputs are never overwritten)'}))
            return 2
    import torch
    torch.set_num_threads(int(os.environ.get('SOUP_THREADS', '3')))
    record = {'tool': TOOL_VERSION, 'what': 'uniform fp32 soup of two P2-architecture checkpoints (SOUP2 rule)',
              'label': args.label, 'utc_start': utc(), 'torch_version': torch.__version__, 'python': sys.executable,
              'threads': torch.get_num_threads(), 'device': 'cpu', 'status': 'RUNNING'}
    timings = {}
    try:
        path_a, path_b = Path(args.a).resolve(), Path(args.b).resolve()
        t = time.time()
        sha_a, sha_b = sha256_file(path_a), sha256_file(path_b)
        timings['sha256_inputs_s'] = round(time.time() - t, 2)
        if args.a_sha256 and sha_a != args.a_sha256:
            raise Stop(f'a sha256 {sha_a} != expected {args.a_sha256}')
        t = time.time()
        payload_a, mmap_a = load_payload(torch, path_a)
        payload_b, mmap_b = load_payload(torch, path_b)
        raw_a, where_a = find_state(torch, payload_a, 'a')
        raw_b, where_b = find_state(torch, payload_b, 'b')
        eff_a, eff_b = fp32_state(torch, raw_a), fp32_state(torch, raw_b)
        timings['load_s'] = round(time.time() - t, 2)
        digest_a, digest_b = state_digest(eff_a), state_digest(eff_b)
        record['inputs'] = {
            'a': {'path': str(path_a), 'sha256': sha_a, 'declared_sha256': args.a_sha256, 'state_found_in': where_a,
                  'mmap': mmap_a, 'meta': meta_of(payload_a), 'fp32_state_digest': digest_a,
                  'fp16_rounded': fp16_rounded(torch, eff_a)},
            'b': {'path': str(path_b), 'sha256': sha_b, 'state_found_in': where_b, 'mmap': mmap_b,
                  'meta': meta_of(payload_b), 'fp32_state_digest': digest_b,
                  'fp16_rounded': fp16_rounded(torch, eff_b)}}
        record['checks'] = compare_structure(torch, raw_a, raw_b, eff_a, eff_b)
        export_report = resolve_export_report(args, path_b, payload_b)
        if export_report is None:
            if args.require_b_export_report:
                raise Stop('no export report found for b (<dir of b>/export/u<updates>/export-report.json); '
                           'give --b-export-report')
            record['b_export'] = {'export_report': None,
                                  'warning': 'b is not linked to an export report: the rule "weights that the '
                                             'export uses" is NOT verified'}
        else:
            record['b_export'] = check_export_link(export_report, path_b, sha_b, digest_b)
        t = time.time()
        used_b, decision = apply_rounding(torch, args, eff_a, eff_b, path_b, payload_b, digest_b)
        timings['rounding_s'] = round(time.time() - t, 2)
        record['rounding'] = decision
        record['inputs']['b']['used_fp32_state_digest'] = state_digest(used_b)
        groups, overall = group_stats(torch, eff_a, used_b, eff_b)
        record['difference_a_b_information'] = {'overall': overall, 'per_group': groups,
                                                'note': 'b after the rounding rule; information only'}
        t = time.time()
        soup = {}
        for key, va in eff_a.items():
            soup[key] = ((va + used_b[key]) * 0.5).contiguous() if va.is_floating_point() else va.clone()
        bad = [k for k, v in soup.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
        if bad:
            raise Stop(f'soup: non-finite values in {bad[:10]}')
        soup_digest = state_digest(soup)
        timings['average_s'] = round(time.time() - t, 2)
        payload = {'kind': 'soup_r10_uniform_fp32_v1', 'dtype': 'float32', 'weights': [0.5, 0.5],
                   'arm': 'soup', 'label': args.label,
                   'members': {'a': {'path': str(path_a), 'sha256': sha_a, **{k: v for k, v in meta_of(payload_a).items()
                                                                              if k in ('kind', 'run', 'arm', 'updates')}},
                               'b': {'path': str(path_b), 'sha256': sha_b, **{k: v for k, v in meta_of(payload_b).items()
                                                                              if k in ('kind', 'run', 'arm', 'updates')}}},
                   'b_rounding': decision['action'], 'rule': args.b_rounding,
                   'architecture': meta_of(payload_a).get('architecture')
                   or 'NativeIntervalATST as parent-p2.pt; load_state_dict strict',
                   'state_digest_fp32': soup_digest, 'tool': TOOL_VERSION, 'utc': utc(), 'model_state': soup}
        del payload_a, payload_b, raw_a, raw_b
        t = time.time()
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(out)
        timings['save_s'] = round(time.time() - t, 2)
        t = time.time()
        again = torch.load(out, map_location='cpu', weights_only=False, mmap=True)
        state_again = again['model_state']
        same = set(state_again) == set(soup) and all(
            state_again[k].dtype == soup[k].dtype and torch.equal(state_again[k], soup[k]) for k in soup)
        if not same:
            raise Stop(f'{out}: the reloaded soup differs from the computed soup')
        record['output'] = {'path': str(out), 'sha256': sha256_file(out), 'bytes': out.stat().st_size,
                            'kind': payload['kind'], 'tensors': len(soup),
                            'floating_tensors': sum(1 for v in soup.values() if v.is_floating_point()),
                            'parameters': int(sum(v.numel() for v in soup.values() if v.is_floating_point())),
                            'dtypes': sorted({str(v.dtype) for v in soup.values()}),
                            'fp32_state_digest': soup_digest, 'reload_equal': True,
                            'loader': "inference/export_frames.py (payload['model_state'], strict)"}
        timings['verify_s'] = round(time.time() - t, 2)
        record['status'] = 'OK'
        code = 0
    except Stop as stop:
        record['status'] = 'STOP'
        record['error'] = str(stop)
        code = 2
    timings['total_s'] = round(time.time() - STARTED, 2)
    record['timings'] = timings
    record['peak_rss_bytes'] = peak_rss_bytes()
    record['peak_rss_gib'] = round(record['peak_rss_bytes'] / 2 ** 30, 3)
    record['utc_end'] = utc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=1, default=str) + '\n')
    summary = {'status': record['status'], 'error': record.get('error'), 'out': str(out) if code == 0 else None,
               'sha256': (record.get('output') or {}).get('sha256'), 'rounding': (record.get('rounding') or {}).get('action'),
               'overall': (record.get('difference_a_b_information') or {}).get('overall'),
               'peak_rss_gib': record['peak_rss_gib'], 'total_s': timings['total_s'], 'report': str(report_path)}
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
