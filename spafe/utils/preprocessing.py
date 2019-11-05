import numpy as np


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

def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.

    Returns:
        array of frames.
        frame length.
    """
    # compute frame length and frame step (convert from seconds to samples)
    frame_length  = win_len * fs
    frame_step    = win_hop * fs
    signal_length = len(sig)

    # Make sure that we have at least 1 frame+
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    frame_length = int(frame_length)
    frame_step = int(frame_step)

    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal_length = num_frames * frame_step + frame_length
    z          = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(sig, z)

    # compute indices
    idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = idx1 + idx2
    frames  = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_length

def windowing(frames, frame_len, win_type="hamming"):
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
    if win_type == "hamming": frames *= np.hamming(frame_len)
    if win_type == "hanning": frames *= np.hanning(frame_len)
    if win_type == "bartlet": frames *= np.bartlet(frame_len)
    if win_type == "kaiser": frames *= np.kaiser(frame_len)
    if win_type == "blackman": frames *= np.blackman(frame_len)
    return frames
