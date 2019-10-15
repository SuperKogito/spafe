"""
based on https://www.researchgate.net/publication/309149564_Robust_Speaker_Verification_Using_GFCC_Based_i-Vectors
"""
import numpy as np
from scipy.fftpack import dct
from ..features import cepstral
from ..utils import processing as proc
from ..fbanks.gammatone_fbanks import gammatone_filter_banks


def gfcc(signal, num_ceps, ceplifter=22):
    """
    Compute GFCC features from an audio signal.

    Args:
         signal  (array) : the audio signal from which to compute features. Should be an N x 1 array
         fs      (int)   : the sampling frequency of the signal we are working with.
         nfilts  (int)   : the number of filters in the filterbank, default 40.
         nfft    (int)   : number of FFT points. Default is 512.
         fl      (float) : lowest band edge of mel filters. In Hz, default is 0.
         fh      (float) : highest band edge of mel filters. In Hz, default is samplerate/2

    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # pre-emphasis -> framing -> windowing -> FFT -> |.|
    pre_emphasised_signal = proc.pre_emphasis(signal)
    frames, frame_length  = proc.framing(pre_emphasised_signal)
    windows               = proc.windowing(frames, frame_length)
    fourrier_transform    = proc.fft(windows)
    abs_fft_values        = np.abs(fourrier_transform)

    #  -> x Gammatone fbanks -> log(.) -> DCT(.)
    gammatone_fbanks_mat = gammatone_filter_banks()
    features             = np.dot(abs_fft_values, gammatone_fbanks_mat.T) # compute the filterbank energies
    nonlin_rect_features = np.power(features, 1/3)
    raw_gfccs            = dct(nonlin_rect_features, type=2, axis=1, norm='ortho')[:,:num_ceps]

    # filter and normalize
    gfccs = proc.lifter(raw_gfccs, ceplifter)
    gfccs = cepstral.cmvn(cepstral.cms(gfccs))
    return gfccs
