"""
based on https://github.com/supikiti/PNCC/blob/master/pncc.py
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
import scipy
import numpy as np
from ..utils.spectral import stft
from ..utils.preprocessing import pre_emphasis
from ..utils.cepstral import cms, cmvn, lifter_ceps
from ..fbanks.gammatone_fbanks import gammatone_filter_banks


def medium_time_power_calculation(power_stft_signal, M=2):
    medium_time_power = np.zeros_like(power_stft_signal)
    power_stft_signal = np.pad(power_stft_signal, [(M, M), (0, 0)], 'constant')
    for i in range(medium_time_power.shape[0]):
        medium_time_power[i, :] = sum([
            1 / float(2 * M + 1) * power_stft_signal[i + k - M, :]
            for k in range(2 * M + 1)
        ])
    return medium_time_power


def asymmetric_lawpass_filtering(rectified_signal, lm_a=0.999, lm_b=0.5):
    floor_level = np.zeros_like(rectified_signal)
    floor_level[0, ] = 0.9 * rectified_signal[0, ]

    for m in range(floor_level.shape[0]):
        x = lm_a * floor_level[m - 1, :] + (1 - lm_a) * rectified_signal[m, :]
        y = lm_b * floor_level[m - 1, :] + (1 - lm_b) * rectified_signal[m, :]
        floor_level[m, :] = np.where(rectified_signal[m, ] >= floor_level[m - 1, :], x, y)
    return floor_level


def temporal_masking(rectified_signal, lam_t=0.85, myu_t=0.2):
    # rectified_signal[m, l]
    temporal_masked_signal = np.zeros_like(rectified_signal)
    online_peak_power = np.zeros_like(rectified_signal)

    temporal_masked_signal[0, :] = rectified_signal[0, ]
    online_peak_power[0, :] = rectified_signal[0, :]


    for m in range(1, rectified_signal.shape[0]):
        online_peak_power[m, :] = np.maximum(lam_t * online_peak_power[m - 1, :], rectified_signal[m, :])
        temporal_masked_signal[m, :] = np.where(rectified_signal[m, :] >= lam_t * online_peak_power[m - 1, :], rectified_signal[m, :], myu_t * online_peak_power[m - 1, :])

    return temporal_masked_signal

def weight_smoothing(final_output, medium_time_power, N=4, L=128):

    spectral_weight_smoothing = np.zeros_like(final_output)
    for m in range(final_output.shape[0]):
        for l in range(final_output.shape[1]):
            l_1 = max(l - N, 1)
            l_2 = min(l + N, L)
            spectral_weight_smoothing[m, l] = (1 / float(l_2 - l_1 + 1)) * \
                sum([(final_output[m, l_] / medium_time_power[m, l_])
                     for l_ in range(l_1, l_2)])
    return spectral_weight_smoothing


def mean_power_normalization(transfer_function,
                             final_output,
                             lam_myu=0.999,
                             L=80,
                             k=1):
    myu = np.zeros(shape=(transfer_function.shape[0]))
    myu[0] = 0.0001
    normalized_power = np.zeros_like(transfer_function)
    for m in range(1, transfer_function.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + \
            (1 - lam_myu) / L * \
            sum([transfer_function[m, s] for s in range(0, L - 1)])
    normalized_power = k * transfer_function / myu[:, None]

    return normalized_power

def medium_time_processing(power_stft_signal, nfilts=22):
    # calculate medium time power
    medium_time_power = medium_time_power_calculation(power_stft_signal)
    lower_envelope = asymmetric_lawpass_filtering(medium_time_power, 0.999,
                                                  0.5)
    subtracted_lower_envelope = medium_time_power - lower_envelope

    # half waverectification
    threshold = 0
    rectified_signal = np.where(subtracted_lower_envelope < threshold,
                                np.zeros_like(subtracted_lower_envelope),
                                subtracted_lower_envelope)


    floor_level = asymmetric_lawpass_filtering(rectified_signal)
    temporal_masked_signal = temporal_masking(rectified_signal)


    # switch excitation or non-excitation
    c = 2
    F = np.where(medium_time_power >= c * lower_envelope,
                 temporal_masked_signal,
                 floor_level)

    # weight smoothing
    spectral_weight_smoothing = weight_smoothing(F, medium_time_power, L=nfilts)
    return spectral_weight_smoothing, F


def pncc(sig,
         fs=16000,
         num_ceps=13,
         nfft=512,
         winlen=0.020,
         winstep=0.010,
         nfilts=26,
         weight_N=4,
         power=2,
         lifter=0,
         low_freq=None,
         high_freq=None,
         dct_type=2,
         use_cmp=True,
         win_len=0.025,
         win_hop=0.01,
         pre_emph=1,
         pre_emph_coeff=0.97,
         normalize=0,
         dither=1,
         sum_power=1,
         band_width=1,
         broaden=0,
         use_energy=False,
         spncc=0):
    """
    Compute the power-normalized cepstral coefficients (SPNCC features) from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        win_type       (float) : window type to apply for the windowing.
                                 Default is hamming.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        dither           (int) : 1 = add offset to spectrum as if dither noise.
                                 Default is 0.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
        bwidth         (float) : width of aud spec filters relative to default.
                                 Default is 1.0.
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_cmp          (int) : apply equal-loudness weighting and cube-root compr.
                                 Default is 0.
        broaden          (int) : flag to retain the (useless?) first and last bands
                                 Default is 0.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        spncc            (int) : if 1 then compute the simple power-normalized
                                 cepstral coefficients (SPNCC).
                                 Default is 0.

    Returns:
        (array) : 2d array of PNCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> STFT()
    stf_trafo, _ = stft(sig, fs)

    #  -> |.|^2
    spectrum_power = np.abs(stf_trafo)**power

    # -> x Filterbanks
    gammatone_filter = gammatone_filter_banks(nfilts=nfilts,
                                              nfft=nfft,
                                              fs=fs,
                                              low_freq=low_freq,
                                              high_freq=high_freq)
    P = np.dot(a=spectrum_power[:, :gammatone_filter.shape[1]],
               b=gammatone_filter.T)

    # medium_time_processing
    S, F = medium_time_processing(P, nfilts=nfilts)

    # time-freq normalization
    T = P * S

    # -> mean power normalization
    U = mean_power_normalization(T, F, L=nfilts)
    # -> power law non linearity
    V = U**(1/15)

    # DCT(.)
    pnccs = scipy.fftpack.dct(V)[:, :num_ceps]

    # liftering
    if lifter > 0:
        pnccs = lifter_ceps(pnccs, lifter)

    # normalization
    if normalize:
        pnccs = cmvn(cms(pnccs))
    return pnccs
