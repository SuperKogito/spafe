"""
based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf
"""
import numpy as np
from scipy.fftpack import dct
from spafe.features import cepstral
import spafe.utils.processing as proc
from spafe.fbanks.mel_fbanks import mel_filter_banks


def mfcc(signal, num_ceps, ceplifter=22):
    """
    Compute MFCC features from an audio signal.
    
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
    
    #  -> x Mel-fbanks -> log(.) -> DCT(.)
    mel_fbanks_mat   = mel_filter_banks()
    features         = np.dot(abs_fft_values, mel_fbanks_mat.T) # compute the filterbank energies
    features_no_zero = proc.zero_handling(features)             # if feat is zero, we get problems with log
    log_features     = np.log(features_no_zero)
    raw_mfccs        = dct(log_features, type=2, axis=1, norm='ortho')[:,:num_ceps]      
    
    # filter and normalize
    mfccs = proc.lifter(raw_mfccs, ceplifter)
    mfccs = cepstral.cmvn(cepstral.cms(mfccs))
    return mfccs

def mfe(signal, fs, frame_length=0.020, frame_stride=0.01, nfilts=40, nfft=512, fl=0, fh=None):
    """
    Compute Mel-filterbank energy features from an audio signal.

    Args:
         signal       (array) : the audio signal from which to compute features. Should be an N x 1 array
         fs           (int)   : the sampling frequency of the signal we are working with.
         frame_length (float) : the length of each frame in seconds.Default is 0.020s
         frame_stride (float) : the step between successive frames in seconds. Default is 0.02s (means no overlap)
         nfilts       (int)   : the number of filters in the filterbank, default 40.
         nfft         (int)   : number of FFT points. Default is 512.
         fl           (float) : lowest band edge of mel filters. In Hz, default is 0.
         fh           (float) : highest band edge of mel filters. In Hz, default is samplerate/2
    
    Returns:
        (array) : features - the energy of fiterbank of size num_frames x num_filters. 
        The energy of each frame: num_frames x 1
    """
    pre_emphasised_signal   = proc.pre_emphasis(signal)
    frames, frame_length    = proc.framing(pre_emphasised_signal, fs)
    windows                 = proc.windowing(frames, frame_length)
    fourrier_transform      = proc.fft(windows)
    power_frames            = proc.power_spectrum(fourrier_transform)

    # compute total energy in each frame
    frame_energies = np.sum(power_frames, 1)

    # Handling zero enegies
    mel_freq_energies = proc.zero_handling(frame_energies)
    return mel_freq_energies