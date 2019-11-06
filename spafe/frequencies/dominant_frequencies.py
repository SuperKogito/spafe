# -*- coding: utf-8 -*-
"""
Reference:
    https://dsp.stackexchange.com/questions/40180/the-exact-definition-of-dominant-frequency
    https://arxiv.org/pdf/1306.0103.pdf
"""
import time
import scipy
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


class DominantFrequenciesExtractor:
    def __init__(self, debug=False):
        self.debug = debug

    def sliding_window(self, sig, fs, window_length_in_ms, skip_in_ms):
        """
        Return slices of a signal based on a sliding winding concept with a
        defined window width and window step.

        Args:
            sig       (array) : audio signal (list of float)
            fs        (int) : sampling rate (= average number of samples pro 1 second)
            window_length_in_ms (int) : slicing window in milli-seconds.
            skip_in_ms          (int) : slicing step in milli-seconds.

        Returns:
            (array) : array window/slice.
        """
        window_length = int(fs / 1000) * window_length_in_ms
        window_skip = int(fs / 1000) * skip_in_ms

        if window_skip is None:
            window_skip = window_length
        for i in range(0, len(sig) - int(window_length), int(window_skip)):
            yield sig[i:i + window_length]

    def split_signal_in_frames(self, sig, fs, frame_size_in_ms=100):
        """
        Return slices of a signal based on a sliding winding concept with a
        fixed window width and window step.

        Args:
            sig       (array) : audio signal (list of float)
            fs        (int) : sampling rate (= average number of samples pro 1 second)
            frame_size_in_ms (int) : slicing window and step in milli-seconds.

        Returns:
            (list) : list of signal slices.
        """
        samples_pro_frame = int(fs / 1000) * frame_size_in_ms
        samples_count_pro_frame = int(
            int(len(sig) / samples_pro_frame) * samples_pro_frame)
        frames = np.split(
            sig[:samples_count_pro_frame],
            len(sig[:samples_count_pro_frame]) /
            int(samples_count_pro_frame))
        try:
            frames.append(sig[samples_count_pro_frame:])
        except BaseException:
            pass
        return frames

    def get_dominant_frequencies(self, slices, fs):
        """
        Returns the dominant audio frequency (in Hertz) of a given audio signal.

        Args:
            slices  (list) : list of signal slices.
            fs     (int) : sampling rate (= average number of samples pro 1 second)

        Returns:
            (array) : array of dominant frequencies.
        """
        dominant_frequencies = []
        for sig in [sig for sig in slices if sig.size > 0]:
            fourrier_transform = np.fft.fft(sig)
            psd = (1 / len(fourrier_transform)) * abs(fourrier_transform)**2
            frequencies = np.fft.fftfreq(sig.size, 1 / fs)
            frequencies = np.array([freq for freq in frequencies if freq >= 0])
            idx = np.argsort(frequencies)
            dominant_frequencies.append(frequencies[idx][np.argmax(psd[idx])])

            # debug: print and plot results
            if self.debug:
                print("Dominant Frequency: %5f KHz" %
                      round(dominant_frequencies[-1] / 1000, 3))
                plt.plot(frequencies[idx] / 1000, psd[idx])
                plt.plot(frequencies[np.argmax(psd[idx])] / 1000,
                         psd[idx].max(), "rx")
                plt.ylabel('Power [dB]')
                plt.xlabel('Frequencies [KHz]')
                plt.ylim(psd.min(), psd.max() + .1 * psd.max())
                plt.xlim(
                    -.25, dominant_frequencies[-1] / 1000 +
                    .25 * dominant_frequencies[-1] / 1000)
        plt.show()
        return np.array(dominant_frequencies)

    def main(self, sig, fs, lower_cutoff=50, upper_cutoff=3000):
        """
        Returns the dominant audio frequencies of a given wave file.

        Args:
            sig (array) : name of an audio file name.

        Returns:
            (array) : array of dominant frequencies.
        """

        # apply Band pass Butterworth filter
        b, a = scipy.signal.butter(6, [(lower_cutoff * 2) / fs,
                                   (upper_cutoff * 2) / fs], 'band')
        w, h = scipy.signal.freqs(b, a, len(sig))
        sig = scipy.signal.lfilter(b, a, sig)

        # compute dominant frequencies
        slices = [w for w in self.sliding_window(sig, fs, 10, 5)]
        dom_freqs = self.get_dominant_frequencies(slices, fs)
        return dom_freqs
