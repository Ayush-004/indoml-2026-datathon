"""Smaller standard audio files with exact float32 waveform preservation."""
from pathlib import Path
import hashlib
import io
import numpy as np
import soundfile as sf


def encode(row):
    source=Path(row['audio_path'])
    original=source.read_bytes()
    assert hashlib.sha256(original).hexdigest()==row['audio_sha256'], 'Source bytes changed before packaging'
    waveform,rate=sf.read(io.BytesIO(original),dtype='float32',always_2d=True)
    assert rate==16000 and waveform.shape[1]==1 and np.isfinite(waveform).all()
    waveform=waveform[:,0]
    assert len(waveform)==round(row['duration']*16000)
    decoded_sha=hashlib.sha256(waveform.astype('<f4',copy=False).tobytes()).hexdigest()
    if row.get('decoded_float32_sha256'):
        assert decoded_sha==row['decoded_float32_sha256']
    subtype=None
    for bits in (16,24):
        scale=2**(bits-1)
        integer=np.rint(waveform.astype('float64')*scale)
        if integer.min()>=-scale and integer.max()<scale and np.array_equal((integer/scale).astype('float32'),waveform):
            subtype='PCM_'+str(bits)
            break
    if subtype is None:
        payload,extension=original,'wav'
        subtype='original_float_waveform'
    else:
        stream=io.BytesIO()
        sf.write(stream,waveform,rate,format='FLAC',subtype=subtype)
        payload,extension=stream.getvalue(),'flac'
    restored,restored_rate=sf.read(io.BytesIO(payload),dtype='float32',always_2d=False)
    assert restored_rate==rate and restored.shape==waveform.shape
    assert hashlib.sha256(restored.astype('<f4',copy=False).tobytes()).hexdigest()==decoded_sha, 'Lossless audio roundtrip changed samples'
    return payload,extension,{'original_audio_sha256':row['audio_sha256'],'original_bytes':len(original),
        'packaged_audio_sha256':hashlib.sha256(payload).hexdigest(),'packaged_bytes':len(payload),
        'decoded_float32_sha256':decoded_sha,'sample_rate':rate,'samples':len(waveform),'subtype':subtype,
        'waveform_bit_exact':True,'resampled':False,'clipped':False}
