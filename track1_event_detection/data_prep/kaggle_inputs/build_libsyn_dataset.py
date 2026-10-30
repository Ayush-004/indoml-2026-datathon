"""build_libsyn_dataset.py: the folder of the Kaggle input dataset "libsyn-v2" = libsyn-audio.zip
(audio/libsyn_v1_016000..047999.flac, stored) + libsyn-manifest.jsonl + libsyn-sessions.json +
val-natural-libcopy-66.txt + reports. No bed file, no validation or test audio.

T1_LIBSYN_ROOT holds v2/ (gen_libsyn.py output, clips 16,000-47,999) and the earlier v1 dataset folder
ds-libsyn-v1/ (it supplies val-natural-libcopy-66.txt: the 66 natural validation clips that contain an exact library
copy; the kernels also report the validation score on the other 1,740 natural clips). The output folder is T1_LIBSYN_ROOT/ds-libsyn-v2/.
"""
import hashlib, json, os, shutil, zipfile
from pathlib import Path
T1 = Path(os.environ.get('T1_LIBSYN_ROOT', 'libsyn'))
SRC, DST = T1 / 'v2', T1 / 'ds-libsyn-v2'
TAGS = ('016000-017999', '018000-047999')
rows, sess = [], {}
for t in TAGS:
    rows += [json.loads(x) for x in open(SRC / f'manifest-{t}.jsonl') if x.strip()]
    sess.update(json.loads((SRC / f'sessions-{t}.json').read_text()))
ids = [r['clip_id'] for r in rows]
assert len(rows) == 32000 and len(set(ids)) == 32000 and set(ids) == set(sess)
idx = sorted(int(i[len('libsyn_v1_'):-4]) for i in ids)
assert idx[0] == 16000 and idx[-1] == 47999 and len(idx) == 32000
with open(DST / 'libsyn-manifest.jsonl', 'w') as f:
    for r in rows:
        assert (SRC / r['audio_path']).is_file()
        f.write(json.dumps(r) + '\n')
(DST / 'libsyn-sessions.json').write_text(json.dumps(sess) + '\n')
shutil.copyfile(T1 / 'ds-libsyn-v1/val-natural-libcopy-66.txt', DST / 'val-natural-libcopy-66.txt')
for t in TAGS:
    shutil.copyfile(SRC / f'gen-report-{t}.json', DST / f'gen-report-{t}.json')
shutil.copyfile(SRC / 'calibration-016000-017999.json', DST / 'calibration-016000-017999.json')
with zipfile.ZipFile(DST / 'libsyn-audio.zip', 'w', compression=zipfile.ZIP_STORED) as z:
    for r in rows:
        z.write(SRC / r['audio_path'], r['audio_path'])
hours = sum(r['duration'] for r in rows) / 3600
(DST / 'README.md').write_text(
    '# libsyn-v2\n\nLIBSYN-v2 training clips for the Track 1 members (rounds L2 and L3).\n'
    f'32,000 clips (generator indices 16,000-47,999), {hours:.2f} h, 16 kHz FLAC PCM_16, in libsyn-audio.zip '
    '(audio/libsyn_v1_XXXXXX.flac).\nThe same generator as v1 (gen_libsyn.py sha256 '
    '959738d7a2639b9f7c9cf36585075237d95238be128b983ddda5ab098690365b, seed 20261006): speech = training-release beds '
    '(DFN3-cleaned, VAD-masked; no val session); noises = the 203-template library and the 3 master recordings from the '
    'organizer validation release. No test audio. Calibration B1-B9 on clips 16,000-17,999: ALL_PASS = True.\n')
(DST / 'dataset-metadata.json').write_text(json.dumps({'title': 'libsyn-v2', 'id': os.environ.get('KAGGLE_DATASET_OWNER', '<owner>') + '/libsyn-v2',
                                                       'licenses': [{'name': 'other'}]}, indent=1) + '\n')
out = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DST.iterdir())}
print(json.dumps({'rows': len(rows), 'hours': round(hours, 3), 'files': out}, indent=1))
