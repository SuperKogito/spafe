from scipy.io.wavfile import read
from spafe.features.rplp import rplp
from spafe.utils.vis import show_features

# read audio
fpath = "../../../test.wav"
fs, sig = read(fpath)

# compute rplps
rplps = rplp(sig,
             fs=fs,
             pre_emph=0,
             pre_emph_coeff=0.97,
             win_len=0.030,
             win_hop=0.015,
             win_type="hamming",
             nfilts=128,
             nfft=1024,
             low_freq=0,
             high_freq=fs/2,
             lifter=0.9,
             normalize="mvn")

# visualize features
show_features(rplps, "Rasta perceptual linear predictions", "PLP Index", "Frame Index")