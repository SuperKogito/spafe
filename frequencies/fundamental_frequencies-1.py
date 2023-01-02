from scipy.io.wavfile import read
from spafe.frequencies.fundamental_frequencies import compute_yin


# read audio
fpath = "../../../tests/data/test.wav"
fs, sig = read(fpath)
duration = len(sig) / fs
harmonic_threshold = 0.85

pitches, harmonic_rates, argmins, times = compute_yin(sig,
                                                      fs,
                                                      win_len=0.050,
                                                      win_hop=0.025,
                                                      low_freq=50,
                                                      high_freq=1000,
                                                      harmonic_threshold=harmonic_threshold)

# xaxis helper function
gen_xaxis_times = lambda v, dt : [float(x) * dt / len(v) for x in range(0, len(v))]


plt.figure(figsize=(14, 12))
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.125, top=0.9, wspace=0.2, hspace=0.99)

# plot audio data
ax1 = plt.subplot(4, 1, 1)
ax1.plot(gen_xaxis_times(sig, duration), sig)
ax1.set_title("Audio data")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time (seconds)")
plt.grid()

# plot F0
ax2 = plt.subplot(4, 1, 2)
ax2.plot(gen_xaxis_times(pitches, duration), pitches)
ax2.set_title("Fundamental frequencies: F0")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time (seconds)")
plt.grid()

# plot Harmonic rate
ax3 = plt.subplot(4, 1, 3, sharex=ax2)
ax3.plot(gen_xaxis_times(harmonic_rates, duration), harmonic_rates, ":o")
ax3.plot(gen_xaxis_times(harmonic_rates, duration), [harmonic_threshold] * len(harmonic_rates), "r:")
ax3.set_title("Harmonic rate")
ax3.set_ylabel("Rate")
ax3.set_xlabel("Time (seconds)")
plt.grid()

# plot Index of minimums of CMND
ax4 = plt.subplot(4, 1, 4, sharex=ax2)
ax4.plot(gen_xaxis_times(argmins, duration), argmins, ":x")
ax4.set_title("Index of minimums of CMND")
ax4.set_ylabel("Frequency (Hz)")
ax4.set_xlabel("Time (seconds)")
plt.grid()
plt.show()