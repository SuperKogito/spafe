"""
based on http://hectordelgado.me/wp-content/uploads/Delgado2016a.pdf
"""
import numpy as np
from scipy.fftpack import dct
from spafe.features import cepstral
import spafe.utils.processing as proc
from spafe.fbanks.mel_fbanks import mel_filter_banks


def cqcc(signal, num_ceps, ceplifter=22, l=5, nfft=512):
    """
    Compute cqcc features from an audio signal.
    
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
    # -> framing -> windowing -> CQT -> |.|^2
    frames, frame_length = proc.framing(signal)
    cq_transform         = proc.cqt(signal)
    cq_values_magnitude  = np.abs(cq_transform)**2

    # -> log(.) -> Uniform sampling -> DCT(.)
    features_no_zero = proc.zero_handling(cq_values_magnitude)
    log_features     = np.log(features_no_zero)
    raw_cqccs        = dct(log_features)[::l]      
    
    # filter and normalize
    cqccs = proc.pre_emphasis(raw_cqccs)
    cqccs = cepstral.cmvn(cepstral.cms(cqccs))
    return cqccs[:num_ceps]    
