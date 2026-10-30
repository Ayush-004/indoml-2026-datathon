"""Track 2 (Noise Event Removal) inference: one enhancement pipeline for every clip, then the mandated ASR.

usage:  python main.py INPUT_DIR OUTPUT_DIR [--system t4|e3] [--device auto|cuda|mps|cpu] [--chunk 100]
                       [--no-asr] [--asr-model ID_OR_DIR] [--ckpt-dir DIR] [--zipenhancer-dir DIR] [--espnet-dir DIR]

Every input WAV (16 kHz mono PCM16, any depth below INPUT_DIR) is enhanced by the same pipeline:
  ZipEnhancer (fine-tuned, whole clip) + TF-GridNet (fine-tuned)  ->  learned time-frequency fusion head  ->  output.
--system t4 (default): the ZipEnhancer output is the average over 4 test-time views of the clip (the clip, the clip
  delayed by 50 samples, the clip with its sign flipped, and both; each output is aligned back and its sign restored
  before the average), and every clip takes the fusion path. An enhancement failure stops the run.
--system e3: one ZipEnhancer view, a route rule (ZipEnhancer alone when at least 20 % of its frames are pass-through
  frames) and a pass-through step (the input, gain-matched, stays where the model changes a frame by less than 20 dB).
Clips longer than 20 s run in 20-s windows with 1-s linear cross-fades. A clip whose output exceeds the int16 range
is scaled down as a whole. Each output WAV has the input file name and the input length.
Then the mandated ASR (ARTPARK-IISc/SraVaani-1.0, unchanged) transcribes the enhanced WAVs -> OUTPUT_DIR/transcripts.jsonl.

The enhancement code is inference/enhance.py with the environment below. It finds its model folders by name below
a work folder; this script links them there from --ckpt-dir, --zipenhancer-dir and --espnet-dir."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM_ENV = {
    't4': {'T2_RAVG_THR': '2.0', 'T2_PP': '', 'T2_NO_RAW_FALLBACK': '1', 'T2_WORKERS': '0', 'T2_ZIP_TTA': 's50f'},
    'e3': {'T2_RAVG_THR': '0.20', 'T2_PP': 'pt20', 'T2_NO_RAW_FALLBACK': '', 'T2_WORKERS': '0', 'T2_ZIP_TTA': ''},
}
CKPT_LINKS = {  # folder name read by inference/enhance.py -> sub-folder of --ckpt-dir and the file it must hold
    'ckpt_zipenhancer': ('zipenhancer_g3asr', 'final.pt'),
    'ckpt_tfgridnet': ('tfgridnet_v8rm', 'final.pt'),
    'ckpt_fusion_head': ('fusion_head', 'head.pt'),
}


def pick_device(want):
    if want != 'auto':
        return want
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--system', choices=sorted(SYSTEM_ENV), default='t4')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--chunk', type=int, default=100, help='clips per fresh runner process (bounds memory growth)')
    ap.add_argument('--no-asr', action='store_true')
    ap.add_argument('--asr-model', default='ARTPARK-IISc/SraVaani-1.0', help='HF id or a local snapshot folder')
    ap.add_argument('--ckpt-dir', default=str(HERE / 'checkpoints'),
                    help='folder with zipenhancer_g3asr/final.pt, tfgridnet_v8rm/final.pt and fusion_head/head.pt')
    ap.add_argument('--zipenhancer-dir', default=str(HERE / 'third_party' / 'zipenhancer'),
                    help='ZipEnhancer code folder that also holds the official zipenhancer_pytorch_model.bin')
    ap.add_argument('--espnet-dir', default=str(HERE / 'third_party' / 'espnet_universal'),
                    help='ESPnet code folder that also holds tfgridnet_urgent25/{config.yaml,valid.loss.ave_5best.pth}')
    a = ap.parse_args()
    inp, out = Path(a.input_dir).resolve(), Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(p for p in inp.rglob('*.wav') if not p.name.startswith('._') and p.parent.name != 'syntheticCleanRefAudio')
    names = [p.name for p in wavs]
    if not wavs or len(names) != len(set(names)):
        sys.exit('no input WAVs, or duplicate file names below INPUT_DIR')
    ckpt = Path(a.ckpt_dir).resolve()
    for sub, fname in CKPT_LINKS.values():
        if not (ckpt / sub / fname).is_file():
            sys.exit(f'missing checkpoint {ckpt / sub / fname} (see checkpoints/README.md)')
    dev = pick_device(a.device)
    work = Path(tempfile.mkdtemp(prefix='t2_'))
    root = work / 'input'
    (root / 'helpers').mkdir(parents=True)          # optional wheels for the runner; empty here (pip installs them)
    (root / 'zipenhancer').symlink_to(Path(a.zipenhancer_dir).resolve())
    (root / 'espnet_universal').symlink_to(Path(a.espnet_dir).resolve())
    for name, (sub, _) in CKPT_LINKS.items():
        (root / name).symlink_to(ckpt / sub)
    print(f'{len(wavs)} clips, system {a.system}, device {dev}, chunk {a.chunk}, work {work}', flush=True)
    for c, i in enumerate(range(0, len(wavs), a.chunk)):
        part = wavs[i:i + a.chunk]
        slug = f'input-c{c:03d}'
        (root / slug / 'clips').mkdir(parents=True)
        for p in part:
            (root / slug / 'clips' / p.name).symlink_to(p)
        cdir = work / slug
        # one scratch folder for all chunks: each chunk unpacks the same packages there again (same bytes,
        # overwritten in place), so the disk use does not grow with the number of chunks
        env = dict(os.environ, **SYSTEM_ENV[a.system], T2_DEVICE=dev, T2_INPUT=str(root), T2_INPUT_SLUG=slug,
                   T2_OUT=str(cdir / 'out'), T2_TMP=str(work / 'tmp'))
        if dev == 'mps':
            env.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.6')
            env.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.35')
        cmd = [sys.executable, str(HERE / 'inference' / 'enhance.py')]
        with open(work / f'{slug}.log', 'w') as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f'chunk {c} failed (exit {r.returncode}); log: {work / (slug + ".log")}')
        with zipfile.ZipFile(cdir / 'out' / 'enhanced_wavs.zip') as z:
            got = [n for n in z.namelist() if n.endswith('.wav')]
            if len(got) != len(part):
                sys.exit(f'chunk {c}: {len(got)} outputs for {len(part)} inputs')
            z.extractall(out)
        print(f'chunk {c}: {len(got)} clips done', flush=True)
    if not a.no_asr:
        r = subprocess.run([sys.executable, str(HERE / 'inference' / 'sravaani_asr.py'), '--wav-dir', str(out), '--out',
                            str(out / 'transcripts.jsonl'), '--model-dir', a.asr_model, '--threads', '8',
                            '--max-per-run', '0'])
        if r.returncode != 0:
            sys.exit(f'ASR failed (exit {r.returncode})')
    shutil.rmtree(work, ignore_errors=True)
    print('done:', out, flush=True)


if __name__ == '__main__':
    main()
