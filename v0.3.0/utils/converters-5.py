import matplotlib.pyplot as plt
from spafe.utils.converters import hz2mel

# generate freqs array -> convert freqs
hz_freqs = [freq for freq in range(0, 8000, 100)]
mel_freqs = [hz2mel(freq) for freq in hz_freqs]

# visualize conversion
plt.figure(figsize=(14,4))
plt.plot(hz_freqs, mel_freqs)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency (Mel)")
plt.title("Hertz to Mel frequency conversion")
plt.tight_layout()
plt.show()