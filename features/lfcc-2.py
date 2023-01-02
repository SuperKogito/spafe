from scipy.io.wavfile import read
from spafe.features.lfcc import lfcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute lfccs
lfccs  = lfcc(sig,
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
show_features(lfccs, "Linear Frequency Cepstral Coefﬁcients", "LFCC Index", "Frame Index")