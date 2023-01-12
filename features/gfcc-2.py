from scipy.io.wavfile import read
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute mfccs and mfes
gfccs  = gfcc(sig,
              fs=fs,
              pre_emph=1,
              pre_emph_coeff=0.97,
              window=SlidingWindow(0.03, 0.015, "hamming"),
              nfilts=128,
              nfft=2048,
              low_freq=0,
              high_freq=8000,
              normalize="mvn")

# visualize features
show_features(gfccs, "Gammatone Frequency Cepstral Coefﬁcients", "GFCC Index", "Frame Index")