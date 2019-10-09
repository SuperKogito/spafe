"""
based on https://asmp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13636-017-0100-x
"""
import numpy as np
from scipy.fftpack import dct
from spafe.features import cepstral
import spafe.utils.processing as proc
from spafe.fbanks.bark_fbanks import bark_filter_banks


def intensity_power_law(w):
    E = ((w**2 + 56.8 * 10**6) * w**4) / ((w**2 + 6.3 * 10**6) * (w**2 + .38 * 10**9) * (w**6 + 9.58 * 10**26))
    return E**(1/3)

def bfcc(signal, num_ceps, ceplifter=22):
    """
    Compute MFCC features from an audio signal.
    CWT : Continuous wavelet transform.
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
    
    #  -> x Bark-fbanks
    bark_fbanks_mat      = bark_filter_banks()
    features             = np.dot(abs_fft_values, bark_fbanks_mat.T)

    # Equal-loudness power law (.) -> Intensity-loudness power law
    ipl_features = intensity_power_law(features)
    
    # -> log(.) -> DCT(.)
    features_no_zero     = proc.zero_handling(ipl_features) # if feat is zero, we get problems with log
    log_features         = np.log(features_no_zero)
    raw_bfccs            = dct(log_features, type=2, axis=1, norm='ortho')[:,:num_ceps]      
    
    # filter and normalize
    bfccs = proc.lifter(raw_bfccs, ceplifter)
    bfccs = cepstral.cmvn(cepstral.cms(bfccs))
    return bfccs