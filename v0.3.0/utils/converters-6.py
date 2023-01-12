import matplotlib.pyplot as plt
from spafe.utils.converters import mel2hz

# generate freqs array -> convert freqs
mel_freqs = [freq for freq in range(0, 8000, 100)]
hz_freqs = [mel2hz(freq) for freq in mel_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(mel_freqs, hz_freqs)
plt.xlabel("Frequency (Mel)")
plt.ylabel("Frequency (Hz)")
plt.title("Mel to Hertz frequency conversion")
plt.tight_layout()
plt.show()