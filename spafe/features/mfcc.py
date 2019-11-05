# -*- coding: utf-8 -*-
"""
Mel Frequency Cepstral Coefficients Extraction
===============================================
"""
import scipy
import numpy as np
from ..utils.spectral import rfft, dct

from ..features.lpc import do_lpc, lpc2cep
from ..utils.cepstral import cms, cmvn, lifter_ceps, spec2cep, cep2spec
from ..fbanks.mel_fbanks import inverse_mel_filter_banks, mel_filter_banks
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from ..utils.spectral import (stft, power_spectrum, powspec, lifter, audspec,
                              postaud, invpostaud, invpowspec, invaudspec)


def mfcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.01,
         win_type="hamming",
         nfilts=26,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1,

         use_cmp=True,
         cep_lifter=22,
         lifter_exp=0.6,
         fb_type='fcmel',

         dither=1,
         sumpower=1,
         band_width=1,
         model_order=0,
         broaden=0):
    """
    Compute MFCC features (Mel-frequency cepstral coefficients) from an audio
    signal. This function offers multiple approaches to features extraction
    depending on the input parameters. Implemenation is using FFT and based on
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf

          - take the absolute value of the FFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components

    Args:
        sig        (array) : a mono audio signal (Nx1) from which to compute features.
        fs           (int) : the sampling frequency of the signal we are working with.
                             Default is 16000.
        nfilts       (int) : the number of filters in the filterbank.
                             Default is 40.
        nfft         (int) : number of FFT points.
                             Default is 512.
        win_time   (float) : window length in sec.
                             Default is 0.025.
        win_hop   (float) : step between successive windows in sec.
                             Default is 0.01.
        num_ceps   (float) : number of cepstra to return.
                             Default is 13.
        lifter_exp (float) : exponent for liftering; 0 = none; < 0 = HTK sin lifter.
                             Default is 0.6.
        sum_power    (int) : 1 = sum abs(fft)^2; 0 = sum abs(fft).
                             Default is 1.
        pre_emph   (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                             Default is 0.97.
        dither       (int) : 1 = add offset to spectrum as if dither noise.
                             Default is 0.
        low_freq     (int) : lowest band edge of mel filters (Hz).
                             Default is 0.
        high_freq     (int) : highest band edge of mel filters (Hz).
                             Default is samplerate / 2 = 8000.
        nbands       (int) : number of warped spectral bands to use.
                             Default is 40.
        bwidth     (float) : width of aud spec filters relative to default.
                             Default is 1.0.
        dct_type     (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                             Default is 2.
        fb_type    ('mel') : frequency warp: 'mel','bark','htkmel','fcmel'.
                             Default is 'mel'.
        use_cmp      (int) : apply equal-loudness weighting and cube-root compr.
                             Default is 0.
        model_order  (int) : if > 0, fit a PLP model of this order.
                             Default is 0.
        broaden      (int) : flag to retain the (useless?) first and last bands
                             Default is 0.
        use_energy   (int) : overwrite C0 with true log energy
                             Default is 0.

    Note:
        The following non-default values nearly duplicate Malcolm Slaney's mfcc
        (i.e. melfcc(d,16000,opts...) =~= log(10)*2*mfcc(d*(2^17),16000) )
            - 'win_time'   : 0.016
            - 'lifter_exp' : 0
            - 'low_freq'   : 133.33
            - 'high_freq'   : 6855.6
            - 'sum_power'  : 0

        The following non-default values nearly duplicate HTK's MFCC
        (i.e. melfcc(d,16000,opts...) =~= 2*htkmelfcc(:,[13,[1:12]])'
        where HTK config has PREEMCOEF = 0.97, NUMCHANS = 20, CEPLIFTER = 22,
        NUMCEPS = 12, WINDOWSIZE = 250000.0, USEHAMMING = T, TARGETKIND = MFCC_0)
            - 'lifter_exp' : -22
            - 'nbands'     : 20
            - 'high_freq'   : 8000
            - 'sum_power'  : 0
            - 'fb_type'    : 'htkmel'
            - 'dct_type'   : 3

    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = np.abs(fourrier_transform)

    #  -> x Mel-fbanks
    mel_fbanks_mat = mel_filter_banks(nfilts=nfilts,
                                        nfft=nfft,
                                        fs=fs,
                                        low_freq=low_freq,
                                        high_freq=high_freq,
                                        scale=scale)
    features = np.dot(abs_fft_values, mel_fbanks_mat.T)

    # -> log(.) -> DCT(.)
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)
    raw_mfccs = dct(x=log_features, type=dct_type, axis=1,
                norm='ortho')[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        raw_mfccs[:, 0] = np.log(mfe(sig=sig,
                          fs=fs,
                          frame_length=win_len,
                          frame_stride=win_hop,
                          nfilts=nfilts,
                          nfft=nfft,
                          fl=low_freq,
                          fh=high_freq))

    # liftering
    if lifter > 0:
        mfccs = lifter_ceps(raw_mfccs, cep_lifter)

    # normalizatio
    if normalize:
        mfccs = cmvn(cms(mfccs))
    return mfccs


def imfcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.01,
         win_type="hamming",
         nfilts=26,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1):
    """
    Compute Inverse MFCC features from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = np.abs(fourrier_transform)

    #  -> x Mel-fbanks -> log(.) -> DCT(.)
    imel_fbanks_mat = inverse_mel_filter_banks(nfilts=nfilts,
                                        nfft=nfft,
                                        fs=fs,
                                        low_freq=low_freq,
                                        high_freq=high_freq,
                                        scale=scale)
    features = np.dot(abs_fft_values, imel_fbanks_mat.T)

    # -> log(.)
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)

    # -> DCT(.)
    imfccs = dct(log_features, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the power
        power_frames = power_spectrum(fourrier_transform)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        imfccs[:, 0] = np.log(energy)

    # apply filter
    if lifter > 0:
        imfcss = lifter_ceps(imfccs, lifter)

    # normalization
    if normalize:
        imfccs = cmvn(cms(imfccs))
    return imfccs


def mfe(sig,
        fs,
        frame_length=0.025,
        frame_stride=0.01,
        nfilts=40,
        nfft=512,
        fl=0,
        fh=None):
    """
    Compute Mel-filterbank energy features from an audio signal.

    Args:
         sig       (array) : the audio signal from which to compute features. Should be an N x 1 array
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
    pre_emphasised_signal = pre_emphasis(sig)
    frames, frame_length = framing(pre_emphasised_signal, fs)
    windows = windowing(frames, frame_length)
    fourrier_transform = rfft(x=windows, n=nfft)
    power_frames = power_spectrum(fourrier_transform)

    # compute total energy in each frame
    frame_energies = np.sum(power_frames, 1)

    # Handling zero enegies
    mel_freq_energies = zero_handling(frame_energies)
    return mel_freq_energies


def melfcc(sig,
           fs=16000,
           low_freq=None,
           high_freq=None,
           num_ceps=13,
           nfilts=40,
           lifter_exp=0.6,
           fb_type='fcmel',
           dct_type=1,
           usecmp=True,
           win_len=0.025,
           win_hop=0.01,
           pre_emph=0.97,
           dither=1,
           sumpower=1,
           band_width=1,
           modelorder=0,
           broaden=0,
           use_energy=False):
    """
    Compute MFCC features (Mel-frequency cepstral coefficients) from an audio
    signal. This function offers multiple approaches to features extraction
    depending on the input parameters.

    Implementation using STFT based on Dan Ellis notes:
    link: http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/mfccs.html

          - take the absolute value of the STFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components

    Args:
        sig     (array) : the audio signal from which to compute features. Should be an N x 1 array
        fs           (int) : the sampling frequency of the signal we are working with.
                             Default is 16000.
        nfilts       (int) : the number of filters in the filterbank.
                             Default is 40.
        nfft         (int) : number of FFT points.
                             Default is 512.
        win_time   (float) : window length in sec.
                             Default is 0.025.
        win_hop   (float) : step between successive windows in sec.
                             Default is 0.01.
        num_ceps   (float) : number of cepstra to return.
                             Default is 13.
        lifter_exp (float) : exponent for liftering; 0 = none; < 0 = HTK sin lifter.
                             Default is 0.6.
        sum_power    (int) : 1 = sum abs(fft)^2; 0 = sum abs(fft).
                             Default is 1.
        pre_emph   (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                             Default is 0.97.
        dither       (int) : 1 = add offset to spectrum as if dither noise.
                             Default is 0.
        low_freq     (int) : lowest band edge of mel filters (Hz).
                             Default is 0.
        high_freq     (int) : highest band edge of mel filters (Hz).
                             Default is samplerate / 2 = 8000.
        nbands       (int) : number of warped spectral bands to use.
                             Default is 40.
        bwidth     (float) : width of aud spec filters relative to default.
                             Default is 1.0.
        dct_type     (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                             Default is 2.
        fb_type    ('mel') : frequency warp: 'mel','bark','htkmel','fcmel'.
                             Default is 'mel'.
        use_cmp      (int) : apply equal-loudness weighting and cube-root compr.
                             Default is 0.
        model_order  (int) : if > 0, fit a PLP model of this order.
                             Default is 0.
        broaden      (int) : flag to retain the (useless?) first and last bands
                             Default is 0.
        use_energy   (int) : overwrite C0 with true log energy
                             Default is 0.

    Note:

        The following non-default values nearly duplicate Malcolm Slaney's mfcc
        (i.e. melfcc(d,16000,opts...) =~= log(10)*2*mfcc(d*(2^17),16000) )
            - 'win_time'   : 0.016
            - 'lifter_exp' : 0
            - 'low_freq'   : 133.33
            - 'high_freq'   : 6855.6
            - 'sum_power'  : 0

        The following non-default values nearly duplicate HTK's MFCC
        (i.e. melfcc(d,16000,opts...) =~= 2*htkmelfcc(:,[13,[1:12]])'
        where HTK config has PREEMCOEF = 0.97, NUMCHANS = 20, CEPLIFTER = 22,
        NUMCEPS = 12, WINDOWSIZE = 250000.0, USEHAMMING = T, TARGETKIND = MFCC_0)
            - 'lifter_exp' : -22
            - 'nbands'     : 20
            - 'high_freq'  : 8000
            - 'sum_power'  : 0
            - 'fb_type'    : 'htkmel'
            - 'dct_type'   : 3

    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    if pre_emph != 0:
        sig = scipy.signal.lfilter(b=[1, -pre_emph], a=1, x=sig)

    pspectrum, logE = powspec(sig,
                              fs=fs,
                              win_len=win_len,
                              win_hop=win_hop,
                              dither=dither)
    aspectrum = audspec(pspectrum,
                        fs=fs,
                        nfilts=nfilts,
                        fb_type=fb_type,
                        low_freq=low_freq,
                        high_freq=high_freq)
    if usecmp:
        aspectrum, _ = postaud(aspectrum, fmax=high_freq, fb_type=fb_type)

    if modelorder > 0:
        lpcas = do_lpc(aspectrum, modelorder)
        cepstra = lpc2cep(lpcas, nout=num_ceps)

    else:
        cepstra, _ = spec2cep(aspectrum, ncep=num_ceps, dct_type=dct_type)

    cepstra = lifter(cepstra, lift=lifter_exp)

    if use_energy:
        cepstra[0, :] = logE

    return cepstra


def invmelfcc(cep,
              fs=16000,
              win_time=0.025,
              win_hop=0.01,
              lifter_exp=0.6,
              sumpower=True,
              pre_emph=0.97,
              high_freq=6500,
              low_freq=50,
              nfilts=40,
              band_width=1,
              dct_type=2,
              fb_type='mel',
              usecmp=False,
              modelorder=0,
              broaden=0,
              excitation=[]):
    """
    Attempt to invert plp cepstra back to a full spectrum and even a waveform.
    x is (noise-excited) time domain waveform; aspc is the
    auditory spectrogram, spec is the |STFT| spectrogram.
    2005-05-15 dpwe@ee.columbia.edu


    Args:
        cep        (array) : the plp ceptra array.
        fs           (int) : the sampling frequency of the signal we are working with.
                             Default is 16000.
        nfilts       (int) : the number of filters in the filterbank.
                             Default is 40.
        nfft         (int) : number of FFT points.
                             Default is 512.
        win_time   (float) : window length in sec.
                             Default is 0.025.
        win_hop   (float) : step between successive windows in sec.
                             Default is 0.01.
        num_ceps   (float) : number of cepstra to return.
                             Default is 13.
        lifter_exp (float) : exponent for liftering; 0 = none; < 0 = HTK sin lifter.
                             Default is 0.6.
        sum_power    (int) : 1 = sum abs(fft)^2; 0 = sum abs(fft).
                             Default is 1.
        pre_emph   (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                             Default is 0.97.
        dither       (int) : 1 = add offset to spectrum as if dither noise.
                             Default is 0.
        low_freq     (int) : lowest band edge of mel filters (Hz).
                             Default is 0.
        high_freq     (int) : highest band edge of mel filters (Hz).
                             Default is samplerate / 2 = 8000.
        nbands       (int) : number of warped spectral bands to use.
                             Default is 40.
        bwidth     (float) : width of aud spec filters relative to default.
                             Default is 1.0.
        dct_type     (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                             Default is 2.
        fb_type    ('mel') : frequency warp: 'mel','bark','htkmel','fcmel'.
                             Default is 'mel'.
        use_cmp      (int) : apply equal-loudness weighting and cube-root compr.
                             Default is 0.
        model_order  (int) : if > 0, fit a PLP model of this order.
                             Default is 0.
        broaden      (int) : flag to retain the (useless?) first and last bands
                             Default is 0.
        use_energy   (int) : overwrite C0 with true log energy
                             Default is 0.

    Note:
        The following non-default values nearly duplicate Malcolm Slaney's mfcc
        (i.e. melfcc(d,16000,opts...) =~= log(10)*2*mfcc(d*(2^17),16000) )
            - 'win_time'   : 0.016
            - 'lifter_exp' : 0
            - 'low_freq'   : 133.33
            - 'high_freq'   : 6855.6
            - 'sum_power'  : 0

        The following non-default values nearly duplicate HTK's MFCC
        (i.e. melfcc(d,16000,opts...) =~= 2*htkmelfcc(:,[13,[1:12]])'
        where HTK config has PREEMCOEF = 0.97, NUMCHANS = 20, CEPLIFTER = 22,
        NUMCEPS = 12, WINDOWSIZE = 250000.0, USEHAMMING = T, TARGETKIND = MFCC_0)
            - 'lifter_exp' : -22
            - 'nbands'     : 20
            - 'high_freq'   : 8000
            - 'sum_power'  : 0
            - 'fb_type'    : 'htkmel'
            - 'dct_type'   : 3
    """
    winpts = int(np.round(win_time * fs))
    nfft = int(np.ceil(np.log(winpts) / np.log(2))**2)
    cep = lifter(cep, lift=lifter_exp, invs=True)

    pspc, _ = cep2spec(cep,
                       nfreq=int(nfilts + 2 * broaden),
                       dct_type=dct_type)

    if usecmp:
        auditory_spectrum, _ = invpostaud(pspc,
                                          fmax=high_freq,
                                          fb_type=fb_type,
                                          broaden=broaden)
    else:
        auditory_spectrum = pspc

    spec, _, _ = invaudspec(auditory_spectrum,
                            fs=fs,
                            nfft=nfft,
                            fb_type=fb_type,
                            low_freq=low_freq,
                            high_freq=high_freq,
                            sumpower=sumpower,
                            band_width=band_width)

    x = invpowspec(spec,
                   fs,
                   win_time=win_time,
                   win_hop=win_hop,
                   excit=excitation)

    if pre_emph != 0:
        x = scipy.signal.lfilter(b=[1, -pre_emph], a=1, x=x)
    return x, auditory_spectrum, spec, pspc
