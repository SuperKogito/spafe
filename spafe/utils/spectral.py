"""

- Description : Spectral utils implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
from scipy.sparse import csr_matrix
from ..utils.preprocessing import windowing


def compute_constant_qtransform(
    frames,
    fs=16000,
    low_freq=0,
    high_freq=None,
    nfft=512,
    number_of_octaves=7,
    number_of_bins_per_octave=12,
    win_type="hamming",
    spectral_threshold=0.0054,
    f0=120,
    q_rate=1.0,
):
    """
    Compute the constant Q-transform as described in [Browm1991]_, [Brown1992]_
    and [Schörkhuber]_.

    Args:
        frames                   (list) : list of audio frames (list of lists).
        fs                        (int) : the sampling frequency of the signal we are working with.
                                          (Default is 16000).
        low_freq                  (int) : lowest band edge of mel filters (Hz).
                                          (Default is 0).
        high_freq                 (int) : highest band edge of mel filters (Hz).
                                          (Default is samplerate / 2).
        nfft                      (int) : number of FFT points.
                                          (Default is 512).
        number_of_octaves         (int) : number of occtaves.
                                          (Default is 7).
        number_of_bins_per_octave (int) : numbers of bins oer occtave.
                                          (Default is 24).
        win_type                (float) : window type to apply for the windowing.
                                          (Default is "hamming").
        spectral_threshold        (int) : spectral threshold.
                                          (Default is 0.005).
        f0                        (int) : fundamental frequency.
                                          (Default is 28).
        q_rate                    (int) : number of FFT points.
                                          (Default is 1.0).

    Returns:
        (numpy.ndarray) : the constant q-transform.

    References:
        .. [Browm1991] : Brown, J. C. (1991). Calculation of a constant Q spectral
                         transform. The Journal of the Acoustical Society of
                         America, 89(1), 425–434. doi:10.1121/1.400476
        .. [Brown1992] : Brown, J. C. & Puckette, M. (1992). "An efficient algorithm
                         for the calculation of a constant Q transform". Journal of the
                         Acoustical Society of America. 92. 2698. 10.1121/1.404385.
        .. [Schörkhuber] : Schörkhuber, C. "Constant-Q transform toolbox for music processing."
                           7th Sound and Music Computing Conference, Barcelona, Spain. 2010.
    """
    high_freq = high_freq or fs / 2

    # calculate the center freqs.
    tmp_cqt_freqs = np.array(
        [
            f0 * 2 ** ((m * number_of_bins_per_octave + n) / number_of_bins_per_octave)
            for m in range(number_of_octaves)
            for n in range(number_of_bins_per_octave)
        ]
    )
    cqt_freqs = tmp_cqt_freqs[tmp_cqt_freqs <= high_freq]

    # calculate Q
    Q = q_rate / (2 ** (1.0 / number_of_bins_per_octave) - 1.0)

    # compute Nks (win_lens)
    win_lens = np.ceil(Q * fs / cqt_freqs).astype(np.int64)
    win_lens = win_lens[win_lens <= nfft]

    # filter center freqs and count number of pitches & frames
    cqt_freqs = cqt_freqs[-1 * len(win_lens) :]
    n_pitch = len(cqt_freqs)
    n_frames = len(frames)

    # calculate kernel
    a = np.zeros((n_pitch, nfft), dtype=np.complex128)
    kernel = np.zeros(a.shape, dtype=np.complex128)

    for k in range(n_pitch):
        Nk = win_lens[k]
        fk = cqt_freqs[k]

        # prepare indices
        start_index = int((nfft - Nk) / 2)
        end_index = start_index + Nk

        # prepare kernel
        temp_a = np.exp(2.0 * np.pi * 1j * (fk / fs) * np.arange(0, Nk))
        a[k, start_index:end_index] = (1 / Nk) * windowing(temp_a, Nk, win_type)
        kernel[k] = np.fft.fft(a[k], nfft)

    # prepare sparse computation vars
    kernel[np.abs(kernel) <= spectral_threshold] = 0.0
    kernel_sparse = csr_matrix(kernel).conjugate() / nfft

    # compute transform
    spec = np.zeros([n_frames, n_pitch], dtype=np.complex128)
    for k, frame in enumerate(frames):
        x = (
            np.r_[frame, np.zeros(nfft - len(frame))]
            if len(frame) < nfft
            else frame[0 : len(frame)]
        )
        spec[k] = np.fft.fft(x, nfft) * kernel_sparse.T
    return spec.T
