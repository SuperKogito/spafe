"""
Created on Sat May 27 15:37:50 2017
Python version of:
D. P. W. Ellis (2009). "Gammatone-like spectrograms", web resource. http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
On the corresponding webpage, Dan notes that he would be grateful if you cited him if you use his work (as above).
This python code does not contain all features present in MATLAB code.
Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017
"""
from __future__ import division
import numpy as np
import scipy.signal as sps


def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100, maxfreq=10000, maxlen=1024):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """

    wts = np.zeros([nfilts, nfft])

    # after Slaney's MakeERBFilters
    EarQ  = 9.26449;
    minBW = 24.7;
    order = 1;

    nFr    = np.array(range(nfilts)) + 1
    em     = EarQ * minBW
    cfreqs = (maxfreq + em) * np.exp(nFr * (-np.log(maxfreq + em) + np.log(minfreq + em)) / nfilts) - em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(nfft / 2 + 1)))
    ucirc = np.exp(1j * 2 * np.pi * ucircArray / nfft);
    # justpoles = 0 :taking out the 'if' corresponding to this.

    ERB = width * np.power(np.power(cfreqs / EarQ, order) + np.power(minBW, order), 1 / order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B / sr)
    theta = 2 * np.pi * cfreqs / sr
    pole = r * np.exp(1j * theta)
    T = 1 / sr
    ebt = np.exp(B * T);
    cpt = 2 * cfreqs * np.pi * T;
    ccpt = 2 * T * np.cos(cpt);
    scpt = 2 * T * np.sin(cpt);
    A11 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2);
    A12 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2);
    A13 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2);
    A14 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2);
    zros = -np.array([A11, A12, A13, A14]) / T;
    wIdx = range(int(nfft / 2 + 1))
    gain = np.abs((-2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
        -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                           np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(
                       2 * cfreqs * np.pi * T))) * (-2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
        -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) + np.sqrt(
        3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) * (
                          -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
                      -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                                  np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(
                              2 * cfreqs * np.pi * T))) * (
                          -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
                      -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                                  np.cos(2 * cfreqs * np.pi * T) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(
                              2 * cfreqs * np.pi * T))) / (
                          -2 / np.exp(2 * B * T) - 2 * np.exp(4 * 1j * cfreqs * np.pi * T) + 2 * (
                          1 + np.exp(4 * 1j * cfreqs * np.pi * T)) / np.exp(B * T)) ** 4);
    # in MATLAB, there used to be 64 where here it says nfilts:
    wts[:, wIdx] = ((T ** 4) / np.reshape(gain, (nfilts, 1))) * np.abs(
        ucirc - np.reshape(zros[0], (nfilts, 1))) * np.abs(ucirc - np.reshape(zros[1], (nfilts, 1))) * np.abs(
        ucirc - np.reshape(zros[2], (nfilts, 1))) * np.abs(ucirc - np.reshape(zros[3], (nfilts, 1))) * (np.abs(
        np.power(np.multiply(np.reshape(pole, (nfilts, 1)) - ucirc, np.conj(np.reshape(pole, (nfilts, 1))) - ucirc),
                 -GTord)));
    wts = wts[:, range(maxlen)];

    return wts, cfreqs


def gammatonegram(x, sr=20000, twin=0.025, thop=0.010, N=64,
                  fmin=50, fmax=10000, width=1.0):
    """
    # Ellis' description in MATLAB:
    # [Y,F] = gammatonegram(X,SR,N,TWIN,THOP,FMIN,FMAX,USEFFT,WIDTH)
    # Calculate a spectrogram-like time frequency magnitude array
    # based on Gammatone subband filters.  Waveform X (at sample
    # rate SR) is passed through an N (default 64) channel gammatone
    # auditory model filterbank, with lowest frequency FMIN (50)
    # and highest frequency FMAX (SR/2).  The outputs of each band
    # then have their energy integrated over windows of TWIN secs
    # (0.025), advancing by THOP secs (0.010) for successive
    # columns.  These magnitudes are returned as an N-row
    # nonnegative real matrix, Y.
    # WIDTH (default 1.0) is how to scale bandwidth of filters
    # relative to ERB default (for fast method only).
    # F returns the center frequencies in Hz of each row of Y
    # (uniformly spaced on a Bark scale).
    # 2009/02/23 DAn Ellis dpwe@ee.columbia.edu
    # Sat May 27 15:37:50 2017 Maddie Cusimano mcusi@mit.edu, converted to python
    """

    # Entirely skipping Malcolm's function, because would require
    # altering ERBFilterBank code as well.
    # i.e., in Ellis' code: usefft = 1
    #assert (x.dtype == 'int16')

    # How long a window to use relative to the integration window requested
    winext = 1;
    twinmod = winext * twin;
    nfft = int(2 ** (np.ceil(np.log(2 * twinmod * sr) / np.log(2))))
    nhop = int(np.round(thop * sr))
    nwin = int(np.round(twinmod * sr))
    [gtm, f] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft / 2 + 1))
    # perform FFT and weighting in amplitude domain
    # note: in MATLAB, abs(spectrogram(X, hanning(nwin), nwin-nhop, nfft, SR))
    #                  = abs(specgram(X,nfft,SR,nwin,nwin-nhop))
    # in python approx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin,
    #                    noverlap=nwin-nhop, nfft=nfft, detrend=False,
    #                    scaling='density', mode='magnitude')
    plotF, plotT, Sxx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin,
                                        noverlap=nwin - nhop, nfft=nfft, detrend=False,
                                        scaling='density', mode='magnitude')
    y = (1 / nfft) * np.dot(gtm, Sxx)

    return y, f

def get_gfcc(audio, rate, log_constant=1e-80, db_threshold=-50.):
    sxx, center_freq = gammatonegram(audio, sr=rate, fmin=20, fmax=int(rate / 2.))
    sxx[sxx == 0] = log_constant
    sxx = 20.0 * np.log10(sxx)  # to db
    sxx[sxx < db_threshold] = db_threshold
    # center_freq
    return sxx
