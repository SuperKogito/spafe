import numpy
import conversion
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt


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

def framing(emphasized_signal, frame_size = 0.025, frame_stride = 0.01):
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

def get_filterbanks(nfilt=20, nfft= 512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq  = highfreq or samplerate/2
    # compute points evenly spaced in mels
    lowmel    = conversion.hz2mel(lowfreq)
    highmel   = conversion.hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin   = numpy.floor((nfft + 1) * conversion.mel2hz(melpoints) / samplerate)
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])

    for j in range(0, nfilt):
        for i in range(int(bin[j]),   int(bin[j+1])): fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])): fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

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


def mfcc(signal, num_ceps, ceplifter=22):
    pre_emphasised_signal  = pre_emphasis(signal)
    frames, frame_length   = framing(pre_emphasised_signal)
    windows                = windowing(frames, frame_length)
    fourrier_transform     = fft(windows)
    power_frames           = power_spectrum(fourrier_transform)
    mel_scale_filter_banks = get_filterbanks()
    mel_scale_filter_banks -= (numpy.mean(mel_scale_filter_banks, axis=0) + 1e-8)
    features               = numpy.dot(power_frames, mel_scale_filter_banks.T)            # compute the filterbank energies
    features               = numpy.where(features == 0, numpy.finfo(float).eps, features) # if feat is zero, we get problems with log
    mfccs                  = dct(features, type=2, axis=1, norm='ortho')[:,:num_ceps]
    mfccs                  = lifter(mfccs, ceplifter)
    mfccs                 -= (numpy.mean(mfccs, axis=0) + 1e-8)
    return mfccs



# setup
sample_rate, signal = scipy.io.wavfile.read('../test.wav')            # File assumed to be in the same directory
mfccs = mfcc(signal, 13)


plt.imshow(mfccs, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('MFCC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()

from python_speech_features import mfcc
mfcc_feat = mfcc(signal,sample_rate)
plt.imshow(mfcc_feat, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('MFCC Coefficient Index')
plt.xlabel('Frame Index')
plt.show()
