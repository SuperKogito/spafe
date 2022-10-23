import matplotlib.pyplot as plt
from spafe.utils.converters import erb2hz

# generate freqs array -> convert freqs
erb_freqs = [freq for freq in range(0, 35, 1)]
hz_freqs = [erb2hz(freq) for freq in erb_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(erb_freqs, hz_freqs)
plt.xlabel("Frequency (Erb)")
plt.ylabel("Frequency (Hz)")
plt.title("Erb to Hertz frequency conversion")
plt.tight_layout()
plt.show()