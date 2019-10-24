import scipy
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from spafe.utils import converters


# init global vars
NFFT = 512

def triangle(x, left, middle, right):
    out = np.zeros(x.shape)
    out[x <= left]   = 0
    out[x >= right]  = 0
    first_half       = np.logical_and(left < x, x <= middle)
    out[first_half]  = (x[first_half] - left) / (middle - left)
    second_half      = np.logical_and(middle <= x, x < right)
    out[second_half] = (right - x[second_half]) / (right - middle)
    return out

def zero_handling(x):
    """
    This function handle the issue with zero values if the are exposed to become
     an argument for any log function.

    Args:
        x: The vector.

    Returns:
        The vector with zeros substituted with epsilon values.
    """
    return np.where(x == 0, np.finfo(float).eps, x)

def pre_emphasis(sig, pre_emph_coeff = 0.97):
    """
    perform preemphasis on the input signal.

    Args:
        signal: The signal to filter.
        coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
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

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames  = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_length

def windowing(frames, frame_len, win_type="hamming"):
    if win_type == "hamming": frames *= np.hamming(frame_len)
    if win_type == "hanning": frames *= np.hanning(frame_len)
    if win_type == "bartlet": frames *= np.bartlet(frame_len)
    if win_type == "kaiser": frames *= np.kaiser(frame_len)
    if win_type == "blackman": frames *= np.blackman(frame_len)
    return frames
