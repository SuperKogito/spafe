# -*- coding: utf-8 -*-
"""
Reference:
    https://dsp.stackexchange.com/questions/40180/the-exact-definition-of-dominant-frequency
    https://arxiv.org/pdf/1306.0103.pdf
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt
from ..utils.spectral import rfft
from ..utils.preprocessing import framing, windowing


def get_dominant_frequencies(sig,
                             fs,
                             butter_filter=False,
                             lower_cutoff=50,
                             upper_cutoff=3000,
                             nfft=512,
                             win_len=0.025,
                             win_hop=0.01,
                             win_type="hamming",
                             debug=False):
    """
    Returns a list of dominant audio frequencies of a given wave file.

    Args:
        sig          (array) : name of an audio file name.
        fs             (int) : sampling rate (= average number of samples pro 1 sec)
        butter_filter (bool) : choose whether to apply a Butterworth filter or not.
                               Default is False.
        lower_cutoff   (int) : filter lower cut-off frequency.
                               Default is 50.
        upper_cutoff   (int) : filter upper cot-off frequency.
                               Default is 3000.
        nfft           (int) : number of FFT points.
                               Default is 512,
        win_len      (float) : window length in sec.
                               Default is 0.025.
        win_hop      (float) : step between successive windows in sec.
                               Default is 0.01.
        win_type     (float) : window type to apply for the windowing.
                               Default is "hamming".
        debug         (bool) : choose whether to plot the results or not.
                               Default is False
    Returns:
        (array) : array of dominant frequencies.
    """
    if butter_filter:
        # apply Band pass Butterworth filter
        b, a = scipy.signal.butter(6, [(lower_cutoff * 2) / fs,
                                       (upper_cutoff * 2) / fs], 'band')
        w, h = scipy.signal.freqs(b, a, len(sig))
        sig = scipy.signal.lfilter(b, a, sig)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)
    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # init dominant frequncies list
    dominant_frequencies = []

    # get dominant frequency for each frame
    for w in windows:
        # compute the fft
        fourrier_transform = rfft(x=w, n=nfft)

        # compute magnitude spectrum
        magnitude_spectrum = (1/nfft) * np.abs(fourrier_transform)
        power_spectrum = (1/nfft)**2 * magnitude_spectrum**2

        # get all frequncies and  only keep positive frequencies
        frequencies = np.fft.fftfreq(len(power_spectrum), 1 / fs)
        frequencies = frequencies[np.where(frequencies >= 0)] // 2 +1

        # keep only half of the spectra
        magnitude_spectrum = magnitude_spectrum[:len(frequencies)]
        power_spectrum = power_spectrum[:len(frequencies)]

        # get id for max spectrum
        idx = np.argmax(power_spectrum)

        # get dom freq and convert it to Hz
        dom_freq = frequencies[idx]

        # add dominant frequency to dominant frequencies list
        dominant_frequencies.append(dom_freq)

    # convert to array, round  and only keep unique values
    dominant_frequencies = np.array(dominant_frequencies)
    dominant_frequencies = np.round(dominant_frequencies, 3)
    dominant_frequencies = np.unique(dominant_frequencies)

    # debugging plot
    if debug:
         plt.plot(frequencies, magnitude_spectrum, "g")
         plt.plot(dominant_frequencies,
                  [magnitude_spectrum[np.where(frequencies == f)] for f in dominant_frequencies],
                  "rx")
         plt.show()

    return dominant_frequencies
