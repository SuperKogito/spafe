#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
based on https://github.com/ilkerbayram/Basic-STFT by ilker bayram
"""
import numpy as np
import matplotlib.pyplot as plt


def pre_process_x(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Prepare window and pad signal audio
    """
    # convert integer to double
    #sig = np.double(sig) / 2.**15

    # STFT parameters
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)

    # compute window
    window = np.hanning(win_length)
    if win_type == "hamm":
        window = np.hamming(win_length)

    # normalization step to ensure that the STFT is self-inverting (or a Parseval frame)
    normalized_window = normalize_window(win=window, hop=hop_length)

    # Compute the STFT
    # zero pad to ensure that there are no partial overlap windows in the STFT computation
    sig = np.pad(sig, (window.size + hop_length, window.size + hop_length),
                 'constant',
                 constant_values=(0, 0))
    return sig, normalized_window, hop_length


def stft(sig, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    sig, normalized_window, hop_length = pre_process_x(sig,
                                                       fs=fs,
                                                       win_type=win_type,
                                                       win_len=win_len,
                                                       win_hop=win_hop)

    X = compute_stft(x=sig, win=normalized_window, hop=hop_length)
    return X, sig


def compute_stft(x, win, hop):
    """
    Compute the short time Fourrier transform of an audio signal x.

    Args:
        x   (array) : audio signal in the time domain
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size

    Returns:
        X : 2d array of the STFT coefficients of x
    """
    # length of the audio signal
    sig_len = x.size

    # length of the window = fft size
    win_len = win.size

    # number of steps to take
    num_steps = (np.ceil((sig_len - win_len) / hop) + 1).astype(int)

    # init STFT coefficients
    X = np.zeros((win_len, num_steps), dtype=complex)

    # normalizing factor
    nf = np.sqrt(win_len)

    for k in range(num_steps - 1):
        d = x[k * hop:k * hop + win_len] * win
        X[:, k] = np.fft.fft(d) / nf

    # the last window may partially overlap with the signal
    d = x[num_steps * hop:]
    X[:, k] = np.fft.fft(d * win[:d.size], n=win_len) / nf
    return X


def istft(X, fs=16000, win_type="hann", win_len=0.025, win_hop=0.01):
    """
    # input variables :
    # X : STFT coefficients
    # win : window to be used for the STFT
    # hop : hop-size
    #
    # output variables :
    # x : inverse STFT of X
    """
    # STFT parameters
    # convert win_len and win_hop from seconds to samples
    win_length = int(win_len * fs)
    hop_length = int(win_hop * fs)

    # compute window
    if win_type == "hann":
        window = np.hanning(win_length)
    elif win_type == "hamm":
        window = np.hamming(win_length)

    # normalization step to ensure that the STFT is self-inverting (or a Parseval frame)
    normalized_window = normalize_window(win=window, hop=hop_length)

    # Compute the ISTFT
    # win_len: length of the window +   num_steps: number of frames
    win_len, num_steps = X.shape[0], X.shape[1]

    # length of the output signal
    sig_len = win_len + (num_steps - 1) * hop_length

    # init output variable
    x = np.zeros((sig_len), dtype=complex)

    # normalizing factor
    nf = np.sqrt(win_len)

    for k in range(num_steps):
        d = nf * np.fft.ifft(X[:, k]) * normalized_window
        x[k * hop_length:k * hop_length + win_len] += d
    return x


def normalize_window(win, hop):
    """
    Normalize the window according to the provided hop-size so that the STFT is
    a tight frame.

    Args:
        win   (int) : window to be used for the STFT
        hop   (int) : hop-size
    """
    N = win.size
    K = int(N / hop)
    win2 = win * win
    z = 1 * win2
    k = 1
    ind1 = N - hop
    ind2 = hop
    while (k < K):
        z[0:ind1] += win2[ind2:N]
        z[ind2:N] += win2[0:ind1]
        ind1 -= hop
        ind2 += hop
        k += 1
    win2 = win / np.sqrt(z)
    return win2


def display_stft(X,
                 fs,
                 len_sig,
                 low_freq=0,
                 high_freq=3000,
                 min_db=-10,
                 max_db=0,
                 normalize=True):
    """
    Plot the stft of an audio signal in the time-frequency plane.

    Args:
        X        (array) : STFT coefficients
        fs         (int) : sampling frequency in Hz (assumed to be integer)
        hop        (int) : hop-size used in the STFT (for labeling the time axis)
        low_freq   (int) : minimun frequency to plot in hz.
                           Default is 0 Hz.
        high_freq  (int) : maximum frequency tp plot in Hz.
                           Default is 3000 Hz.
        min_db     (int) : minimun magnitude to display in dB
                           Default is 0 dB.
        max_db     (int) : maximum magnitude to display in dB.
                           Default is -10 dB.
        normalize (bool) : Normalize input.
                           Default is True.
    """
    # normalize : largest coefficient magnitude is unity
    X_temp = X.copy()
    if normalize:
        X_temp /= np.amax(abs(X_temp))

    # compute frequencies array
    Freqs = np.array([low_freq, high_freq])
    Fd = (Freqs * X_temp.shape[0] / fs).astype(int)

    # compute values matrix
    Z = X_temp[Fd[1]:Fd[0]:-1, :]
    Z = np.clip(np.log(np.abs(Z) + 1e-50), min_db, max_db)
    Z = 255 * (Z - min_db) / (max_db - min_db)

    # compute duration
    time = float(len_sig) / float(fs)

    # plotting
    plt.imshow(Z,
               extent=[0, time, low_freq / 1000, high_freq / 1000],
               aspect="auto")
    plt.ylabel('Frequency (Khz)')
    plt.xlabel('Time (sec)')
    plt.show()
