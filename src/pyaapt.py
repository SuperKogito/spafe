""" This script implements YAAPT algorithm. More information can be found
    here: http://harvey.binghamton.edu/~hhu1/paper/Zahorian2008spectral.pdf """

import scipy
import numpy as np
from scipy.signal import butter
from scipy.signal import lfilter

import pylab as pl


# Runs the YAAPT algorithm to pull out fundamental frequencies for each time
def yaapt(data, sample_rate=44100):
    # Separate out the signal into the regular and the non-linearly
    # transformed signal, afterwards bandpass filtering both.
    #signal, signal_nonlinear = data, data ** 2
    signal, signal_nonlinear = stft(data).T, stft(data).T ** 2

    # Other stuff...
    return 0


# Implements special harmonics correlation. This can likely be optimized.
# Input: M x N matrix, with M being frequencies and N being time.
#   a window_length in frequencies.
# Output: M x N matrix, with M being frequencies and N being time.
def shc(signal, window_length=40., number_harmonics=3):
    # This function finds what frequency corresponds to what index
    freq_idx = lambda f: int(np.floor(float(f) / (44100. / 1024.)))

    # This will hold the output
    shc_signal = np.zeros(signal.shape)

    # Loop through the window length
    for f in [-1, 0, 1]:
        for freq in range(signal.shape[0]):
            # Calculate the indices for the harmonics to consider
            harmonics = [r*freq+f for r in range(number_harmonics)
                         if r*freq+f < signal.shape[0]]
            # Compute the product of the amplitudes of this frequency
            # multiplied by the amplitudes of the harmonics.
            h = np.prod(np.array([signal[h,:] for h in harmonics]), axis=0)
            shc_signal[freq, :] += h

    return shc_signal


# Runs STFT on the data for use in examining the spectrogram
# Output: M x N matrix, M being time and N being frequencies
def stft(data, fftsize=1024, overlap=4):   
    w = scipy.hanning(fftsize + 1)[:-1]
    r = [np.fft.rfft(w*data[i:i+fftsize]) for i in range(0, len(data)-fftsize, int(fftsize / overlap))]
    return np.array(r)


#####################################################################################
if __name__ == '__main__':
    import pylab as pl
    from scipy.io import wavfile

    # Read in a WAV file of a Saxophone playing A220
    sample_rate, wav_data = wavfile.read('sample2.wav')
    print('Sample Rate: %s' % sample_rate)

    # Pull out the fundamental frequency from the signal
    freqs = yaapt(wav_data, sample_rate)