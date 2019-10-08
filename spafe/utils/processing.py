import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from spafe.utils import converters


# init global vars
NFFT = 512

def triangle(x, left, middle, right):
    out = numpy.zeros(x.shape)
    out[x <= left]   = 0
    out[x >= right]  = 0
    first_half       = numpy.logical_and(left < x, x <= middle)
    out[first_half]  = (x[first_half] - left) / (middle - left)
    second_half      = numpy.logical_and(middle <= x, x < right)
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
    return numpy.where(x == 0, numpy.finfo(float).eps, x)

def pre_emphasis(signal, pre_emphasis_coeff = 0.97):
    """
    perform preemphasis on the input signal.

    Args:
        signal: The signal to filter.
        coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])

def framing(emphasized_signal, sample_rate=8000, frame_size = 0.025, frame_stride = 0.01):
    # compute frame length and frame step
    frame_length  = frame_size   * sample_rate
    frame_step    = frame_stride * sample_rate  # Convert from seconds to samples

    signal_length = len(emphasized_signal)
    frame_length  = int(round(frame_length))
    frame_step    = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z          = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames  = pad_signal[indices.astype(numpy.int32, copy=False)]
    return frames, frame_length

def windowing(frames, frame_length):
    frames *= numpy.hamming(frame_length)
    return frames

def fft(frames, nfft = NFFT):
    return numpy.fft.rfft(frames, nfft)

def power_spectrum(fourrier_transform,  nfft = NFFT):
    magnitude_frames = numpy.absolute(fourrier_transform)          # Magnitude of the FFT
    power_frames     = ((1.0 / nfft) * ((magnitude_frames) ** 2))  # Power Spectrum
    return power_frames

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n    = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra