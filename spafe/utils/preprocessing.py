import numpy as np
import scipy.ndimage
from spafe.utils.spectral import rfft
from .exceptions import ParameterError, ErrorMsgs


def zero_handling(x):
    """
    handle the issue with zero values if they are exposed to become an argument
    for any log function.

    Args:
        x (array): input vector.

    Returns:
        vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)


def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.

    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])


def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.

    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.

    Returns:
        blocked/framed array.
    """
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,
                                           shape=(nrows, stride_length),
                                           strides=(stride_step*n, n))


def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames (=Frame blocking).

    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        win_len (float) : window length in sec.
                          Default is 0.025.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.01.

    Returns:
        array of frames.
        frame length.

    Notes:
    ------
        Uses the stride trick to accelerate the processing.
    """
    # run checks and assertions
    if win_len < win_hop:
        raise ParameterError(ErrorMsgs["win_len_win_hop_comparison"])

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # make sure to use integers as indices
    frames = stride_trick(sig, int(frame_length), int(frame_step))
    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(frames[-1], np.array([0]*(frame_length - len(frames[0]))))

    return frames, frame_length


def windowing(frames, frame_len, win_type="hamming", beta=14):
    """
    generate and apply a window function to avoid spectral leakage.

    Args:
        frames  (array) : array including the overlapping frames.
        frame_len (int) : frame length.
        win_type  (str) : type of window to use.
                          Default is "hamming"

    Returns:
        windowed frames.
    """
    if   win_type == "hamming" : windows = np.hamming(frame_len)
    elif win_type == "hanning" : windows = np.hanning(frame_len)
    elif win_type == "bartlet" : windows = np.bartlett(frame_len)
    elif win_type == "kaiser"  : windows = np.kaiser(frame_len, beta)
    elif win_type == "blackman": windows = np.blackman(frame_len)
    windowed_frames = frames * windows
    return windowed_frames


def remove_silence(sig, fs, win_len=0.25, win_hop=0.25, threshold=-35):
    """
    generate and apply a window function to avoid spectral leakage.

    Args:
        frames  (array) : array including the overlapping frames.
        frame_len (int) : frame length.
        win_type  (str) : type of window to use.
                          Default is "hamming"

    Returns:
        windowed frames.
    """
    # framing
    frames, frames_len = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # compute short time energies to get voiced frames
    amplitudes = np.abs(rfft(frames, len(frames)))
    energy =  np.sum(amplitudes, axis=-1) / len(frames)**2
    energy =  10 * np.log10(zero_handling(energy))

    # normalize energy to 0 dB then filter and format
    energy = energy - energy.max()
    energy = scipy.ndimage.filters.median_filter(energy, 5)
    energy = np.repeat(energy, frames_len)

    # compute vad and get speech frames
    vad = np.array(energy > threshold, dtype=sig.dtype)
    vframes = np.array(frames.flatten()[np.where(vad==1)], dtype=sig.dtype)
    return energy, vad, np.array(vframes, dtype=np.float64)
