from scipy.io.wavfile import read
from spafe.features.lpc import lpcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute lpccs
lpccs = lpcc(sig,
             fs=fs,
             pre_emph=0,
             pre_emph_coeff=0.97,
             window=SlidingWindow(0.03, 0.015, "hamming"))

# visualize features
show_features(lpccs, "Linear Prediction Cepstral Coefﬁcients", "LPCCs Index","Frame Index")