"""
based on:
    - paper  : Perceptual linear predictive (PLP) analysis of speech 
    - author : Hynek Hermansky
    - link   : https://pdfs.semanticscholar.org/b578/f4faeb00b808e8786d897447f2493b12b4e9.pdf
"""
"""
based on https://www.researchgate.net/profile/Namrata_Dave2/publication/261914482_Feature_extraction_methods_LPC_PLP_and_MFCC_in_speech_recognition/links/562dce4908ae04c2aeb4aa1b/Feature-extraction-methods-LPC-PLP-and-MFCC-in-speech-recognition.pdf
"""
import numpy as np
from spafe.features import cepstral
import spafe.utils.processing as proc
import spafe.utils.levinsondr as ldr
from spafe.fbanks.bark_fbanks import bark_filter_banks


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

def plp(signal, num_ceps, ceplifter=22):
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
    frames, frame_length  = proc.framing(signal)
    windows               = proc.windowing(frames, frame_length)
    fourrier_transform    = proc.fft(windows)
    abs_fft_values        = np.abs(fourrier_transform)
    
    #  -> x Bark-fbanks
    bark_fbanks_mat = bark_filter_banks()
    features        = np.dot(abs_fft_values, bark_fbanks_mat.T)

    # Equal-loudness power law (.) -> Intensity-loudness power law
    pre_emphasised_feats = proc.pre_emphasis(features)
    ipl_features         = intensity_power_law(pre_emphasised_feats)
    
    # -> IDFT(.)
    idft_features = proc.ifft(ipl_features)

    # -> linear prediction  analysis -> cepstral analysis        
    lp_features = lp(idft_features)
    raw_plps    = cepstral_analysis(lp_features)      
    
    # reshape
    s, x = padding_factor(raw_plps, 13)
    raw_plps = (np.append(raw_plps, x*[0])).reshape(s, 13)
    
    # normalize
    plps = proc.lifter(raw_plps, ceplifter)
    plps = cepstral.cmvn(cepstral.cms(plps))
    return plps