import numpy as np
from spafe.utils.converters import erb2hz
from spafe.utils.vis import show_fbanks
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks

# init var
fs = 8000
nfilts = 7
nfft = 1024
low_freq = 0
high_freq = fs / 2

# compute freqs for xaxis
ghz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
    # gamma fbanks
    gamma_fbanks_mat, gamma_freqs = gammatone_filter_banks(nfilts=nfilts,
                                                           nfft=nfft,
                                                           fs=fs,
                                                           low_freq=low_freq,
                                                           high_freq=high_freq,
                                                           scale=scale,
                                                           order=4)
    # visualize filter bank
    show_fbanks(
        gamma_fbanks_mat,
        [erb2hz(freq) for freq in gamma_freqs],
        ghz_freqs,
        label + "Gamma Filter Bank",
        ylabel="Weight",
        x1label="Frequency / Hz",
        x2label="Frequency / erb",
        figsize=(14, 5),
        fb_type="gamma")