"""
from  https://github.com/notwa/dsp/blob/master/lib/wav.py
"""
import numpy as np
from .util import lament, count_channels


def wav_smart_read(fn):
    lament('wav_smart_read(): DEPRECATED; use wav_read instead.')
    # don't use this, it fails to load good files.
    import scipy.io.wavfile as wav
    srate, s = wav.read(fn)
    if s.dtype != np.float64:
        bits = s.dtype.itemsize*8
        s = np.asfarray(s)/2**(bits - 1)
    return srate, s


def wav_smart_write(fn, srate, s):
    lament('wav_smart_write(): DEPRECATED; use wav_write instead.')
    import scipy.io.wavfile as wav
    si = np.zeros_like(s, dtype='int16')
    bits = si.dtype.itemsize*8
    si += np.clip(s*2**(bits - 1), -32768, 32767)
    wav.write(fn, srate, si)


def wav_read(fn):
    import ewave
    with ewave.open(fn) as f:
        s = f.read()
        srate = f.sampling_rate
    if s.dtype == np.float32:
        s = np.asfarray(s)
    elif s.dtype != np.float64:
        bits = s.dtype.itemsize*8
        s = np.asfarray(s)/2**(bits - 1)
    return s, srate


def wav_write(fn, s, srate, dtype='h'):
    import ewave
    if dtype in ('b', 'h', 'i', 'l') and np.max(np.abs(s)) > 1:
        lament('wav_write(): WARNING; clipping')
    with ewave.open(fn, 'w', srate, dtype, count_channels(s)) as f:
        f.write(s)
