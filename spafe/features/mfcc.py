# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
"""
based on http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf
"""
import numpy as np
from scipy.fftpack import dct
from ..features import cepstral
from ..utils import processing as proc
from ..fbanks.mel_fbanks import inverse_mel_filter_banks, mel_filter_banks


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
    features         = np.dot(abs_fft_values, mel_fbanks_mat.T)
    features_no_zero = proc.zero_handling(features)
    log_features     = np.log(features_no_zero)
    raw_mfccs        = dct(log_features, type=2, axis=1, norm='ortho')[:,:num_ceps]

    # filter and normalize
    mfccs = proc.lifter(raw_mfccs, ceplifter)
    mfccs = cepstral.cmvn(cepstral.cms(mfccs))
    return mfccs


def imfcc(signal, num_ceps, ceplifter=22):
    """
    Compute Inverse MFCC features from an audio signal.

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
    imel_fbanks_mat  = inverse_mel_filter_banks()
    features         = np.dot(abs_fft_values, imel_fbanks_mat.T)
    features_no_zero = proc.zero_handling(features)
    log_features     = np.log(features_no_zero)
    raw_imfccs       = dct(log_features, type=2, axis=1, norm='ortho')[:,:num_ceps]

    # filter and normalize
    imfccs = proc.lifter(raw_imfccs, ceplifter)
    imfccs = cepstral.cmvn(cepstral.cms(imfccs))
    return imfccs

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
