import numpy as np
from spafe.utils.converters import bark2hz
from spafe.utils.vis import show_fbanks
from spafe.fbanks.bark_fbanks import bark_filter_banks

# init var
fs = 8000
nfilt = 7
nfft = 1024
low_freq = 0
high_freq = fs / 2

# compute freqs for xaxis
bhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
    # bark fbanks
    bark_fbanks_mat, bark_freqs = bark_filter_banks(nfilts=nfilt,
                                                    nfft=nfft,
                                                    fs=fs,
                                                    low_freq=low_freq,
                                                    high_freq=high_freq,
                                                    scale=scale)

    # visualize filter bank
    show_fbanks(
        bark_fbanks_mat,
        [bark2hz(freq) for freq in bark_freqs],
        bhz_freqs,
        label + "Bark Filter Bank",
        ylabel="Weight",
        x1label="Frequency / Hz",
        x2label="Frequency / bark",
        figsize=(14, 5),
        fb_type="bark",
    )