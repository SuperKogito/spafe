import matplotlib.pyplot as plt
from spafe.utils.converters import hz2bark

# generate freqs array -> convert freqs
hz_freqs = [freq for freq in range(0, 8000, 10)]
bark_freqs = [hz2bark(freq) for freq in hz_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(hz_freqs, bark_freqs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency (Bark)")
plt.title("Hertz to Bark scale frequency conversion")
plt.tight_layout()
plt.show()