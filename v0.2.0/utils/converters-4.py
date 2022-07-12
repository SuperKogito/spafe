import matplotlib.pyplot as plt
from spafe.utils.converters import bark2hz

# generate freqs array -> convert freqs
bark_freqs = [freq for freq in range(0, 80, 5)]
hz_freqs = [bark2hz(freq) for freq in bark_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(bark_freqs, hz_freqs)
plt.xlabel("Frequency (Bark)")
plt.ylabel("Frequency (Hz)")
plt.title("Bark to Hertz frequency conversion")
plt.tight_layout()
plt.show()