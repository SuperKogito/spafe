from scipy.io.wavfile import read
from spafe.features.lpc import lpcc
from spafe.utils.vis import show_features

# read audio
fpath = "../../../test.wav"
fs, sig = read(fpath)

# compute lpccs
lpccs = lpcc(sig,
             fs=fs,
             pre_emph=0,
             pre_emph_coeff=0.97,
             win_len=0.030,
             win_hop=0.015,
             win_type="hamming")

# visualize features
show_features(lpccs, "Linear Prediction Cepstral Coefﬁcients", "LPCCs Index","Frame Index")