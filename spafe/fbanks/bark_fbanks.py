import numpy as np
from _fbank import Filterbank


# Bark frequency scale
def hz2bark(f):
    """
    Convert Hz frequencies to Bark.
    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].
    Returns
    -------
    z : numpy array
        Frequencies in Bark [Bark].
    """
    raise NotImplementedError('please check this function, it produces '
                              'negative values')
    # TODO: use Zwicker's formula?
    #       return 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500.) ** 2)
    return (26.81 / (1. + 1960. / np.asarray(f))) - 0.53


def bark2hz(z):
    """
    Convert Bark frequencies to Hz.
    Parameters
    ----------
    z : numpy array
        Input frequencies [Bark].
    Returns
    -------
    f : numpy array
        Frequencies in Hz [Hz].
    """
    raise NotImplementedError('please check this function, it produces weird '
                              'values')
    # TODO: use Zwicker's formula? what's the inverse of the above?
    return 1960. / (26.81 / (np.asarray(z) + 0.53) - 1.)


def bark_frequencies(fmin=20., fmax=15500.):
    """
    Returns frequencies aligned on the Bark scale.
    Parameters
    ----------
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    Returns
    -------
    bark_frequencies : numpy array
        Frequencies with Bark spacing [Hz].
    """
    # frequencies aligned to the Bark-scale
    frequencies = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                            1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                            4400, 5300, 6400, 7700, 9500, 12000, 15500])
    # filter frequencies
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # return
    return frequencies


def bark_double_frequencies(fmin=20., fmax=15500.):
    """
    Returns frequencies aligned on the Bark-scale.
    The list also includes center frequencies between the corner frequencies.
    Parameters
    ----------
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    Returns
    -------
    bark_frequencies : numpy array
        Frequencies with Bark spacing [Hz].
    """
    # frequencies aligned to the Bark-scale, also includes center frequencies
    frequencies = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                            510, 570, 630, 700, 770, 840, 920, 1000, 1080,
                            1170, 1270, 1370, 1480, 1600, 1720, 1850, 2000,
                            2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700,
                            4000, 4400, 4800, 5300, 5800, 6400, 7000, 7700,
                            8500, 9500, 10500, 12000, 13500, 15500])
    # filter frequencies
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # return
    return frequencies

# helper functions for filter creation
def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.
    Parameters
    ----------
    frequencies : numpy array
        Input frequencies [Hz].
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].
    unique_bins : bool, optional
        Return only unique bins, i.e. remove all duplicate bins resulting from
        insufficient resolution at low frequencies.
    Returns
    -------
    bins : numpy array
        Corresponding (unique) bins.
    Notes
    -----
    It can be important to return only unique bins, otherwise the lower
    frequency bins can be given too much weight if all bins are simply summed
    up (as in the spectral flux onset detection).
    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to the closest bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # only keep unique bins if requested
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    return indices



# border definitions of the 24 critical bands of hearing
bark = [100,   200,  300,  400,  510,  630,   770,   920,
        1080, 1270, 1480, 1720, 2000, 2320,  2700,  3150,
        3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

eq_loudness = np.array([[ 55,   40,  32,  24,  19,  14, 10,  6,  4,  3,  2,
                           2,    0,  -2,  -5,  -4,   0,  5, 10, 14, 25, 35],
                        [ 66,   52,  43,  37,  32,  27, 23, 21, 20, 20, 20,
                          20,   19,  16,  13,  13,  18, 22, 25, 30, 40, 50],
                        [ 76,   64,  57,  51,  47,  43, 41, 41, 40, 40, 40,
                        39.5,   38,  35,  33,  33,  35, 41, 46, 50, 60, 70],
                        [ 89,   79,  74,  70,  66,  63, 61, 60, 60, 60, 60,
                          59,   56,  53,  52,  53,  56, 61, 65, 70, 80, 90],
                        [103,   96,  92,  88,  85,  83, 81, 80, 80, 80, 80,
                          79,   76,  72,  70,  70,  75, 79, 83, 87, 95,105],
                        [118,  110, 107, 105, 103, 102,101,100,100,100,100,
                          99,   97,  94,  90,  90,  95,100,103,105,108,115]])

loudn_freq = np.array([31.62,    50,  70.7,   100, 141.4,   200, 316.2,  500,
                       707.1,  1000,  1414,  1682,  2000,  2515,  3162, 3976,
                        5000,  7071, 10000, 11890, 14140, 15500])



def bark_filter_banks(nfilt=20, nfft= 512, samplerate=16000, lowfreq=0, highfreq=None):
    # calculate bark-filterbank
    loudn_bark = np.zeros((eq_loudness.shape[0], len(bark)))
    i, j       = 0, 0

    for bsi in bark:

        while j < len(loudn_freq) and bsi > loudn_freq[j]:
            j += 1

        j -= 1

        if np.where(loudn_freq == bsi)[0].size != 0: # loudness value for this frequency already exists
            loudn_bark[:,i] = eq_loudness[:,np.where(loudn_freq == bsi)][:,0,0]
        else:
            w1 = 1 / np.abs(loudn_freq[j] - bsi)
            w2 = 1 / np.abs(loudn_freq[j + 1] - bsi)
            loudn_bark[:,i] = (eq_loudness[:,j]*w1 + eq_loudness[:,j+1]*w2) / (w1 + w2)

        i += 1
    return loudn_bark


fbanks = bark_filter_banks()
import matplotlib.pyplot as plt 
plt.plot(fbanks)