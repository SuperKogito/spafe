"""
baesd on https://www.researchgate.net/publication/326893017_Novel_Spectral_Root_Cepstral_Features_for_Replay_Spoof_Detection
"""
import numpy as np
from scipy.fftpack import dct
from ..features import cepstral
from ..utils import processing as proc
from ..fbanks.mel_fbanks import mel_filter_banks


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
    abs_fft_values        = np.abs(fourrier_transform)**2

    #  -> x Mel-fbanks -> log(.) -> DCT(.)
    mel_fbanks_mat   = mel_filter_banks()
    features         = np.dot(abs_fft_values, mel_fbanks_mat.T)
    features_no_zero = proc.zero_handling(features)
    log_features     = np.log(features_no_zero)
    raw_mfccs        = dct(log_features, type=2, axis=1, norm='ortho')[:,:num_ceps]

    # filter and normalize
    mfccs = proc.lifter(raw_mfccs, ceplifter)
    mfccs = cepstral.cmvn(cepstral.cms(mfccs))
    return mfccs
