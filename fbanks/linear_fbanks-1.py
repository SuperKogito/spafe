from spafe.utils.vis import show_fbanks
from spafe.fbanks.linear_fbanks import linear_filter_banks


# init var
fs = 8000
nfilt = 7
nfft = 1024
low_freq = 0
high_freq = fs / 2

# compute freqs for xaxis
lhz_freqs = np.linspace(low_freq, high_freq, nfft //2+1)

for scale, label in [("constant", ""), ("ascendant", "Ascendant "), ("descendant", "Descendant ")]:
    # ascendant linear fbank
    linear_fbanks_mat, lin_freqs = linear_filter_banks(nfilts=nfilt,
                                                       nfft=nfft,
                                                       fs=fs,
                                                       low_freq=low_freq,
                                                       high_freq=high_freq,
                                                       scale=scale)

    # visualize fbanks
    show_fbanks(
        linear_fbanks_mat,
        lin_freqs,
        lhz_freqs,
        label + "Linear Filter Bank",
        ylabel="Weight",
        x1label="Frequency / Hz",
        figsize=(14, 5),
        fb_type="lin")