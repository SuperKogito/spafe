from spafe.features.spfeats import extract_feats
from scipy.io.wavfile import read
from pprint import pprint

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute erb spectrogram
spectral_feats = extract_feats(sig, fs)
pprint(spectral_feats)