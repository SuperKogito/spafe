import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from spafe.utils import functions
from spafe.utils import fbanks


def mfcc(signal, num_ceps, ceplifter=22):
    pre_emphasised_signal  = pre_emphasis(signal)
    frames, frame_length   = framing(pre_emphasised_signal)
    windows                = windowing(frames, frame_length)
    fourrier_transform     = fft(windows)
    power_frames           = power_spectrum(fourrier_transform)
    mel_scale_filter_banks = fbanks.mel_filter_banks()
    features               = numpy.dot(power_frames, mel_scale_filter_banks.T)            # compute the filterbank energies
    features               = numpy.where(features == 0, numpy.finfo(float).eps, features) # if feat is zero, we get problems with log
    mfccs                  = dct(features, type=2, axis=1, norm='ortho')[:,:num_ceps]
    mfccs                  = lifter(mfccs, ceplifter)
    mfccs                 -= (numpy.mean(mfccs, axis=0) + 1e-8)
    return mfccs

def mfe(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,
        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):
    """
    Compute Mel-filterbank energy features from an audio signal.

    Args:
         signal (array)          : the audio signal from which to compute features.
                                   Should be an N x 1 array
         sampling_frequency (int): the sampling frequency of the signal
                                   we are working with.
         frame_length (float)  : the length of each frame in seconds.
                                 Default is 0.020s
         frame_stride (float)  : the step between successive frames in seconds.
                                 Default is 0.02s (means no overlap)
         num_filters (int)     : the number of filters in the filterbank,
                                 default 40.
         fft_length (int)      : number of FFT points. Default is 512.
         low_frequency (float) : lowest band edge of mel filters.
                                 In Hz, default is 0.
         high_frequency (float): highest band edge of mel filters.
                                 In Hz, default is samplerate/2
    Returns:
        array: features - the energy of fiterbank of size num_frames x num_filters.
               The energy of each frame: num_frames x 1
    """
    pre_emphasised_signal   = pre_emphasis(signal)
    frames, frame_length    = framing(pre_emphasised_signal)
    windows                 = windowing(frames, frame_length)
    fourrier_transform      = fft(windows)
    power_frames            = power_spectrum(fourrier_transform)

    # calculation of the power sprectum
    coefficients = power_spectrum.shape[1]

    # this stores the total energy in each frame
    frame_energies = np.sum(power_spectrum, 1)

    # Handling zero enegies.
    mel_freq_energies = functions.zero_handling(frame_energies)
    return mel_freq_energies




# setup
sample_rate, signal = scipy.io.wavfile.read('test.wav')            # File assumed to be in the same directory
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
