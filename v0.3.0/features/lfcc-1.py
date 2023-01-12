from spafe.features.lfcc import linear_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from scipy.io.wavfile import read

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute spectrogram
lSpec, lfreqs = linear_spectrogram(sig,
                                   fs=fs,
                                   pre_emph=0,
                                   pre_emph_coeff=0.97,
                                   window=SlidingWindow(0.03, 0.015, "hamming"),
                                   nfilts=128,
                                   nfft=2048,
                                   low_freq=0,
                                   high_freq=fs/2)

# visualize spectrogram
show_spectrogram(lSpec.T,
                 fs,
                 xmin=0,
                 xmax=len(sig)/fs,
                 ymin=0,
                 ymax=(fs/2)/1000,
                 dbf=80.0,
                 xlabel="Time (s)",
                 ylabel="Frequency (kHz)",
                 title="Linear spectrogram (dB)",
                 cmap="jet")