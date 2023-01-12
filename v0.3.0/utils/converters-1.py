import matplotlib.pyplot as plt
from spafe.utils.converters import hz2erb

# generate freqs array -> convert freqs
hz_freqs = [freq for freq in range(0, 8000, 10)]
erb_freqs = [hz2erb(freq) for freq in hz_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(hz_freqs, erb_freqs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency (Erb)")
plt.title("Hertz to Erb scale frequency conversion")
plt.tight_layout()
plt.show()