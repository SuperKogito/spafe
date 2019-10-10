"""
paper:
"""
import numpy as np
from spafe.features import cepstral
import spafe.utils.processing as proc
import spafe.utils.levinsondr as ldr
from spafe.fbanks.mel_fbanks import mel_filter_banks


def padding_factor(vec, j):
    s = vec.shape[0] // j  + 1
    f = np.abs(s * j - vec.shape[0]) 
    return s, f 

def cepstral_analysis(feats):
    """
    Do cepstral analysis.

    from: https://pdfs.semanticscholar.org/0b44/265790c6008622c0c3de2aa1aea3ca2e7762.pdf
        >> Cepstral coefficients are obtained from the predictor coefficients 
        >> by a recursion that is equivalent to the logarithm of the model 
        >> spectrum followed by an inverse Fourier transform
    """
    feats_spectrum   = np.fft.fft(feats)
    features_no_zero = proc.zero_handling(feats_spectrum)
    log_features     = np.log(features_no_zero)
    return np.abs(np.fft.ifft(log_features))

def lp(vec):
    a, G, eps = ldr.lev_durb(vec)
    return a

def intensity_power_law(w):
    E = ((w**2 + 56.8 * 10**6) * w**4) / ((w**2 + 6.3 * 10**6) * (w**2 + .38 * 10**9) * (w**6 + 9.58 * 10**26))
    return E**(1/3)

def rplp(signal, num_ceps, ceplifter=22):
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
    
    #  -> x Mel-fbanks
    mel_fbanks_mat = mel_filter_banks()
    features       = np.dot(fourrier_transform, mel_fbanks_mat.T)

    # -> IDFT(.)
    idft_features = proc.ifft(features)

    # -> linear prediction  analysis -> cepstral analysis        
    lp_features = lp(idft_features)
    raw_rplps    = cepstral_analysis(lp_features)      
    
    # reshape
    s, x = padding_factor(raw_rplps, 13)
    raw_rplps = (np.append(raw_rplps, x*[0])).reshape(s, 13)
    
    # normalize
    rplps = proc.lifter(raw_rplps, ceplifter)
    rplps = cepstral.cmvn(cepstral.cms(rplps))
    return rplps