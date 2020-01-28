# -*- coding: utf-8 -*-
"""
This module is part of the spafe library and has the purpose of of computing the following spectral stats:
    - meanfreq : mean frequency (in kHz)
    - sd       : standard deviation of frequency
    - median   : median frequency (in kHz)
    - Q25      : first quantile (in kHz)
    - Q75      : third quantile (in kHz)
    - IQR      : interquantile range (in kHz)
    - skew     : skewness (see note in specprop description)
    - kurt     : kurtosis (see note in specprop description)
    - sp.ent   : spectral entropy
    - sfm      : spectral flatness
    - mode     : mode frequency
    - centroid : frequency centroid (see specprop)
    - peakf    : peak frequency (frequency with highest energy)
    - meanfun  : average of fundamental frequency measured across acoustic signal
    - minfun   : minimum fundamental frequency measured across acoustic signal
    - maxfun   : maximum fundamental frequency measured across acoustic signal
    - meandom  : average of dominant frequency measured across acoustic signal
    - mindom   : minimum of dominant frequency measured across acoustic signal
    - maxdom   : maximum of dominant frequency measured across acoustic signal
    - dfrange  : range of dominant frequency measured across acoustic signal
    - modindx  : modulation index. Calculated as the accumulated absolute difference
                 between adjacent measurements of fundamental frequencies divided
                 by the frequency range
    - label    : male or female

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

Reference:
    http://ijeee.iust.ac.ir/article-1-1074-en.pdf
"""
import scipy
import numpy as np
from ..utils.spectral import stft, rfft
from ..frequencies.dominant_frequencies import get_dominant_frequencies
from ..frequencies.fundamental_frequencies import FundamentalFrequenciesExtractor


def compute_fund_freqs(sig, fs):
    """
    compute fundamental frequencies.

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # fundamental frequencies calculations
    fund_freqs_extractor = FundamentalFrequenciesExtractor(debug=False)
    pitches, harmonic_rates, argmins, times = fund_freqs_extractor.main(
        sig=sig, fs=fs)
    return pitches


def compute_dom_freqs_and_mod_index(sig,
                                    fs,
                                    lower_cutoff=50,
                                    upper_cutoff=3000,
                                    nfft=512,
                                    win_len=0.03,
                                    win_hop=0.015,
                                    win_type='hamming',
                                    debug=False):
    """
    compute dominant frequencies and modulation index.

    Args:
        sig (array) : spectral centroid.
        fs (int) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # dominant frequencies calculations
    dom_freqs = get_dominant_frequencies(sig=sig,
                                         fs=fs,
                                         lower_cutoff=50,
                                         upper_cutoff=upper_cutoff,
                                         nfft=nfft,
                                         win_len=win_len,
                                         win_hop=win_hop,
                                         win_type=win_type,
                                         debug=debug)

    # modulation index calculation
    changes = np.abs(dom_freqs[:-1] - dom_freqs[1:])
    dfrange = dom_freqs.max() - dom_freqs.min()
    if dom_freqs.min() == dom_freqs.max():
        mod_index = 0
    else:
        mod_index = changes.mean() / dfrange
    return dom_freqs, mod_index


def spectral_centroid(sig, fs):
    """
    compute spectral centroid.
    """
    # compute magnitude spectrum
    magnitude_spectrum = np.fft.rfft(sig)
    # compute positive frequencies
    freqs = np.abs(np.fft.fftfreq(len(sig), 1.0 / fs)[:len(sig) // 2 + 1])
    # return weighted mean
    sc = np.sum(magnitude_spectrum * freqs) / np.sum(magnitude_spectrum)
    return sc


def spectral_flatness(sig):
    """
    compute spectral flatness.
    """
    # compute magnitude spectrum
    magnitude_spectrum = np.fft.rfft(sig)
    # select half of the spectrum due to symetrie
    magnitude_spectrum = magnitude_spectrum[:len(sig) // 2 + 1]
    sf = scipy.stats.mstats.gmean(magnitude_spectrum) / np.mean(
        magnitude_spectrum)
    return sf


def spectral_rolloff(sig, fs, k=0.85):
    # convert to frequency domain
    magnitude_spectrum, _ = stft(sig=sig, fs=fs)
    power_spectrum = np.abs(magnitude_spectrum)**2
    tbins, fbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    tstamps = (np.arange(0, tbins - 1) * (tbins / float(fs)))
    # compute the spectral sum
    spectral_sum = np.sum(power_spectrum, axis=1)

    # find frequency-bin indeces where the cummulative sum of all bins is higher
    # than k-percent of the sum of all bins. Lowest index = Rolloff
    sr = [
        np.where(np.cumsum(power_spectrum[t, :]) >= k * spectral_sum[t])[0][0]
        for t in range(tbins - 1)
    ]
    sr = np.asarray(sr).astype(float)

    # convert frequency-bin index to frequency in Hz
    sr = (sr / fbins) * (fs / 2.0)
    return sr, np.asarray(tstamps)


def spectral_flux(sig, fs):
    # convert to frequency domain
    magnitude_spectrum, _ = stft(sig=sig, fs=fs)
    tbins, fbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    tstamps = (np.arange(0, tbins - 1) * (tbins / float(fs)))
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2,
                        axis=1)) / fbins

    return sf[1:], np.asarray(tstamps)


def spectral_spread(centroid, spectrum, fs):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    bin_count, numerator, denominator = 0, 0, 0

    for bin_i in spectrum:
        # Compute center frequency
        f = ((fs / 2.0) / len(spectrum)) * bin_count
        numerator = numerator + (((f - centroid)**2) * abs(bin_i))
        denominator = denominator + abs(bin_i)
        bin_count = bin_count + 1

    return np.sqrt((numerator * 1.0) / denominator)


def zero_crossing_rate(sig, fs, block_length=256):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(sig) / block_length))

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0, num_blocks - 1) * (block_length / float(fs)))
    zcr = []

    for i in range(0, num_blocks - 1):
        start = i * block_length
        stop = np.min([(start + block_length - 1), len(sig)])

        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(sig[start:stop]))))
        zcr.append(zc)

    return np.asarray(zcr), np.asarray(timestamps)


def root_mean_square(sig, fs, block_length=256):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(sig) / block_length))

    # when do these blocks begin (time in seconds)?
    tstamps = (np.arange(0, num_blocks - 1) * (block_length / float(fs)))

    rms = []

    for i in range(0, num_blocks - 1):

        start = i * block_length
        stop = np.min([(start + block_length - 1), len(sig)])

        # This is wrong but why? rms_seg = np.sqrt(np.mean(sig[start:stop]**2))
        rms_seg = np.sqrt(np.mean(np.power(sig[start:stop], 2)))
        rms.append(rms_seg)
    return np.asarray(rms), np.asarray(tstamps)


def spectral_bandwidth(sig, fs):
    return []


def extract_feats(sig, fs, nfft=512):
    """
    Compute the spectral features.

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # init features dictionary
    feats = {}

    # compute the fft
    fourrier_transform = rfft(sig, nfft)

    # compute magnitude spectrum
    magnitude_spectrum = (1/nfft) * np.abs(fourrier_transform)
    power_spectrum = (1/nfft)**2 * magnitude_spectrum**2

    # get all frequncies and  only keep positive frequencies
    frequencies = np.fft.fftfreq(len(power_spectrum), 1 / fs)
    frequencies = frequencies[np.where(frequencies >= 0)] // 2 + 1

    # keep only half of the spectra
    magnitude_spectrum = magnitude_spectrum[:len(frequencies)]
    power_spectrum = power_spectrum[:len(frequencies)]

    # define amplitudes and spectrum
    spectrum = power_spectrum
    amplitudes = power_spectrum
    amp_cumsum = np.cumsum(amplitudes)

    # general stats
    feats["duration"] = len(sig) / float(fs)
    feats["spectrum"] = spectrum

    # spectral stats I
    feats["mean_frequency"] = frequencies.sum()
    feats["peak_frequency"] = frequencies[np.argmax(amplitudes)]
    feats["frequencies_std"] = frequencies.std()
    feats["amplitudes_cum_sum"] = np.cumsum(amplitudes)
    feats["mode_frequency"] = frequencies[amplitudes.argmax()]
    feats["median_frequency"] = np.median(frequencies)
    feats["frequencies_q25"] = frequencies[len(amp_cumsum[amp_cumsum <= 0.25])-1]
    feats["frequencies_q75"] = frequencies[len(amp_cumsum[amp_cumsum <= 0.75])-1]
    feats["iqr"] = feats["frequencies_q75"] - feats["frequencies_q25"]

    # spectral stats II
    feats["freqs_skewness"] = scipy.stats.skew(frequencies)
    feats["freqs_kurtosis"] = scipy.stats.kurtosis(frequencies)
    feats["spectral_entropy"] = scipy.stats.entropy(amplitudes)
    feats["spectral_flatness"] = spectral_flatness(sig)
    feats["spectral_centroid"] = spectral_centroid(sig, fs)
    feats["spectral_bandwidth"] = spectral_bandwidth(sig, fs)
    feats["spectral_spread"] = spectral_spread(feats["spectral_centroid"],
                                               feats["spectrum"], fs)
    feats["spectral_flatness"] = spectral_flatness(sig)
    feats["spectral_rolloff"] = spectral_rolloff(sig, fs)

    # compute energy
    feats["energy"] = magnitude_spectrum

    # compute root-mean-square (RMS).
    feats["rms"] = root_mean_square(sig=sig, fs=fs)

    # compute the zero-crossing rate of an audio time series
    feats["zcr"] = zero_crossing_rate(sig=sig, fs=fs)

    # spectral stats
    feats["spectral_mean"] = np.mean(spectrum)
    feats["spectral_rms"] = np.sqrt(np.mean(spectrum**2))
    feats["spectral_std"] = np.std(spectrum)
    feats["spectral_variance"] = np.var(spectrum)

    # assign fundamental frequencies stats
    fund_freqs = compute_fund_freqs(sig=sig, fs=fs)
    feats["meanfun"] = fund_freqs.mean()
    feats["minfun"] = fund_freqs.min()
    feats["maxfun"] = fund_freqs.max()

    # assign dominant frequencies stats
    dom_freqs, mod_idx = compute_dom_freqs_and_mod_index(sig=sig,
                                                         fs=fs,
                                                         lower_cutoff = 50,
                                                         upper_cutoff = 3000,
                                                         nfft = 512,
                                                         win_len = 0.03,
                                                         win_hop = 0.015,
                                                         win_type = 'hamming',
                                                         debug = False)
    feats["meandom"] = dom_freqs.mean()
    feats["mindom"] = dom_freqs.min()
    feats["maxdom"] = dom_freqs.max()

    # range of dominant frequency measured across acoustic signal
    feats["dfrange"] = feats["maxdom"] - feats["mindom"]

    # modulation index: Calculated as the accumulated absolute difference
    # between adjacent measurements of fundamental frequencies divided by the
    # frequency range
    feats["modindex"] = mod_idx
    return feats
