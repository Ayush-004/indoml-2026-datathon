"""Build the int16 audio stores of this study in ./cache (one pass, idempotent).
    python w00_stores.py valsyn valclean valnat train testsyn
The train store is the seeded null sample of wlib.train_null_sample (natural training-release clips)."""
import sys
import zipfile

import wlib


def main(names):
    for name in names:
        if name in ('valsyn', 'valclean', 'valnat'):
            meta = wlib.val_meta()
            with zipfile.ZipFile(wlib.VAL_ZIP) as z:
                if name == 'valsyn':
                    items = [(r['segmentFileName'], wlib.zip_loader(z, 'validation/syntheticNoiseAudio/' + r['segmentFileName']))
                             for r in meta if r['syntheticData']]
                elif name == 'valclean':
                    items = [(r['segmentFileName'], wlib.zip_loader(z, 'validation/syntheticCleanRefAudio/' + r['segmentFileName']))
                             for r in meta if r['syntheticData']]
                else:
                    items = [(r['segmentFileName'], wlib.zip_loader(z, 'validation/naturalNoisyAudio/' + r['segmentFileName']))
                             for r in meta if not r['syntheticData']]
                idx = wlib.build_store(name, items)
        elif name == 'train':
            sample = wlib.train_null_sample()
            items = [(cid, wlib.file_loader(wlib.TRAIN_DIR / ap)) for cid, ap, q, d in sample]
            idx = wlib.build_store(name, items)
        elif name == 'testsyn':
            clips = set(wlib.test_syn_clips())
            with zipfile.ZipFile(wlib.TEST_ZIP) as z:
                members = {}
                for n in z.namelist():
                    if n.endswith('.wav') and '__MACOSX' not in n:
                        b = n.rsplit('/', 1)[-1]
                        if b in clips:
                            members[b] = n
                assert len(members) == len(clips), (len(members), len(clips))
                items = [(c, wlib.zip_loader(z, members[c])) for c in sorted(clips)]
                idx = wlib.build_store(name, items)
        else:
            raise SystemExit('unknown store ' + name)
        tot = sum(v[1] for v in idx.values())
        print(name, len(idx), 'clips', round(tot / wlib.SR / 3600, 3), 'h', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
