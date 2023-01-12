# -*- coding: utf-8 -*-
"""

- Description : Implementation Dominant Frequency Extraction Using the YIN-Algorithm.
- Copyright (c) 2019-2023 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy import signal

from ..utils.preprocessing import framing, windowing, WindowType


def get_dominant_frequencies(
    sig: np.ndarray,
    fs: int,
    butter_filter: bool = False,
    lower_cutoff: float = 50,
    upper_cutoff: float = 3000,
    nfft: int = 512,
    win_len: float = 0.025,
    win_hop: float = 0.01,
    win_type: WindowType = "hamming",
    only_positive: bool = True,
) -> np.ndarray:
    """
    Returns a list of dominant audio frequencies of a given wave file based
    on [Rastislav]_ and [Luca]_.

    Args:
        sig          (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                     (int) : sampling rate (= average number of samples pro 1 sec)
        butter_filter         (bool) : choose whether to apply a Butterworth filter or not.
                                       (Default is False).
        lower_cutoff         (float) : filter lower cut-off frequency.
                                       (Default is 50).
        upper_cutoff         (float) : filter upper cot-off frequency.
                                       (Default is 3000).
        nfft                   (int) : number of FFT points.
                                       (Default is 512).
        win_len              (float) : window length in sec.
                                       (Default is 0.025).
        win_hop              (float) : step between successive windows in sec.
                                       (Default is 0.01).
        win_type               (str) : window type to apply for the windowing.
                                       (Default is "hamming").
        only_positive         (bool) : if True then returns only positive frequncies.
                                       (Default is True).

    Returns:
        (numpy.ndarray) : array of dominant frequencies.

    References:
        .. [Rastislav] : Rastislav T. (2013). Dominant Frequency Extraction. CoRR, abs/1306.0103.
        .. [Luca] : Luca, The exact definition of dominant frequency? https://dsp.stackexchange.com/a/40183/37123

    Examples:

        .. plot::

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
    """
    if butter_filter:
        # apply Band pass Butterworth filter
        b, a = signal.butter(
            6, [(lower_cutoff * 2) / fs, (upper_cutoff * 2) / fs], "band"
        )
        sig = signal.lfilter(b, a, sig)

    # -> framing
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # init dominant frequencies list
    dominant_frequencies = []

    # get dominant frequency for each frame
    for w in windows:
        # compute the fft
        fourrier_transform = np.fft.rfft(a=w, n=nfft)

        # compute magnitude spectrum
        magnitude_spectrum = (1 / nfft) * np.abs(fourrier_transform)
        power_spectrum = (1 / nfft) * magnitude_spectrum**2

        # get all frequncies and  only keep positive frequencies
        frequencies = np.fft.fftfreq(len(power_spectrum), 1 / fs)

        # get id for max spectrum
        idx = np.argmax(power_spectrum)

        # get dom freq and convert it to Hz
        dom_freq = frequencies[idx]

        # add dominant frequency to dominant frequencies list
        dominant_frequencies.append(dom_freq)

    # convert to array, round  and only keep unique values
    dominant_frequencies = np.array(dominant_frequencies)
    dominant_frequencies = np.round(dominant_frequencies, 3)

    # filter out negative frequncies
    if only_positive:
        dominant_frequencies = dominant_frequencies[np.where(dominant_frequencies >= 0)]

    return dominant_frequencies
