"""Loader for the P2 model (parent-p2.pt) on the Mac.

This module only points sys.path at the model sources (atst_red/: the indoml package, context_crop.py,
mixed_training.py; the native bundle for PretrainedSED and the ATST-F / E5 assets) and then calls the
original context_crop.load_parent(). Inputs: T1_NATIVE_BUNDLE (native bundle folder), T1_P2_CHECKPOINT
(parent-p2.pt).
"""
from pathlib import Path
import hashlib
import json
import sys

import os

ATST_RED = Path(__file__).resolve().parents[2] / 'atst_red'
NATIVE = Path(os.environ.get('T1_NATIVE_BUNDLE', 'native-bundle'))
PARENT = Path(os.environ.get('T1_P2_CHECKPOINT', 'parent-p2.pt'))
PARENT_SHA = '419c441e911419b33fa066d67ffa43bb63a8c6b3c659175ccb565f0ecfb83373'
PARENT_STATE = '3fa453a39509b804f80e628c53643c346824373b63c81b0deb5d877bc14a64e9'

# P2's organizer decoder (threshold 0.65).
ORGANIZER_DECODER = {'threshold': .65, 'nms_iou': 0., 'min_duration': .08}
CATEGORIES = ('animal', 'vehicle_traffic', 'baby_child', 'singing_music',
              'phone_signal_alarm', 'appliance_machine', 'human_non_speech')

for entry in (str(NATIVE / 'source'), str(ATST_RED)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def api_module():
    from indoml import atst_intervals as api
    return api


def load_p2(device='cpu', head_device=None):
    """Return (model, metadata). `model` is the native NativeIntervalATST."""
    import context_crop
    api = api_module()
    model, metadata = context_crop.load_parent(api, NATIVE, PARENT, device)
    if head_device is not None and str(head_device) != str(device):
        import torch
        model.set_device(torch.device(device), torch.device(head_device))
    assert api.state_digest(model.state_dict()) == PARENT_STATE
    return model, metadata


def row_for(clip_id, audio_path, duration):
    """The exact inference row shape evaluation.load_organizer() builds."""
    return {'clip_id': clip_id, 'audio_path': str(audio_path), 'duration': duration,
            'num_frames': (round(duration * 16000) + 639) // 640,
            'frame_hop': .04, 'frame_offset': .02, 'feature_dim': 768}


def packs(rows, cap=4):
    """gold_event_fit.packs: at most `cap` native 10 s windows per forward."""
    current, windows = [], 0
    for row in rows:
        count = 1 + max(0, (row['num_frames'] - 250 + 124) // 125)
        assert count <= cap, 'clip exceeds the native window cap'
        if current and windows + count > cap:
            yield current
            current, windows = [], 0
        current.append(row)
        windows += count
    if current:
        yield current


def predict_rows(model, rows, decoder=None, cap=4):
    """The organizer inference path: infer_outputs -> candidate_intervals -> decode."""
    import torch
    api = api_module()
    decoder = dict(ORGANIZER_DECODER if decoder is None else decoder)
    model.eval()
    model.frequency_augmentation = None
    out = []
    with torch.inference_mode():
        for selected in packs(rows, cap):
            for row, values in api.infer_outputs(model, selected, len(selected)):
                candidates = api.candidate_intervals(row, values['presence'], values['distances'])
                out.append(api.decode_candidates(row, candidates, decoder))
    return out


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]
