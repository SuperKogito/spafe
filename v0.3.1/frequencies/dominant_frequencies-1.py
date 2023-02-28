import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies

# init vars
nfft = 512
win_len = 0.020
win_hop = 0.010

# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)

# compute dominant frequencies
dominant_frequencies = get_dominant_frequencies(sig,
                                                fs,
                                                butter_filter=False,
                                                lower_cutoff=0,
                                                upper_cutoff=fs/2,
                                                nfft=nfft,
                                                win_len=win_len,
                                                win_hop=win_hop,
                                                win_type="hamming")

# compute FFT, Magnitude, Power spectra
fourrier_transform = np.absolute(np.fft.fft(sig, nfft))
magnitude_spectrum = fourrier_transform[:int(nfft / 2) + 1]
power_spectrum = (1.0 / nfft) * np.square(fourrier_transform)
power_spectrum = 20*np.log10(power_spectrum)
freqs = np.fft.rfftfreq(power_spectrum.size, 1/fs)
idx = np.argsort(freqs)

# plot
fmin = 500
fmax = 1500

y = power_spectrum
x = freqs
idx = np.argsort(freqs)

plt.figure(figsize=(14, 4))
plt.plot(x[idx], y[idx], "g")
plt.axis((fmin-10, fmax+10, 0, max(y)*(1.1)))

for i, dom_freq in enumerate(np.unique(dominant_frequencies)):
    if  (fmin < dom_freq < fmax):
        plt.vlines(x=dom_freq, ymin=0, ymax=max(y), colors="red", linestyles=":")
        plt.text(dom_freq, max(y) , "({:.1f})".format(dom_freq))

plt.grid()
plt.title("Dominant frequencies (Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum amplitude")
plt.show()