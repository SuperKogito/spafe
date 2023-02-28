from scipy.io.wavfile import read
from spafe.features.lpc import lpc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute lpcs
lpcs, _ = lpc(sig,
              fs=fs,
              pre_emph=0,
              pre_emph_coeff=0.97,
              window=SlidingWindow(0.030, 0.015, "hamming"))

# visualize features
show_features(lpcs, "Linear prediction coefficents", "LPCs Index", "Frame Index")