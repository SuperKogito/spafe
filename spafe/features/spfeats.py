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
    - modindx  : modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
    - label    : male or female

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

Reference:
    http://ijeee.iust.ac.ir/article-1-1074-en.pdf
"""
import scipy
import numpy as np
from ..utils.spectral import stft
from ..utils import preprocessing as proc
from ..frequencies.dominant_frequencies import DominantFrequenciesExtractor
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
    print(sig, fs)
    fund_freqs_extractor = FundamentalFrequenciesExtractor(debug=False)
    pitches, harmonic_rates, argmins, times = fund_freqs_extractor.main(
        sig=sig, fs=fs)
    return pitches


def compute_dom_freqs_and_mod_index(sig, fs):
    """
    compute dominant frequencies and modulation index.

    Args:
        sig (array) : spectral centroid.
        fs (int) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # dominant frequencies calculations
    dom_freqs_extractor = DominantFrequenciesExtractor()
    dom_freqs = dom_freqs_extractor.main(sig=sig, fs=fs)

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
    freqs  = np.abs(np.fft.fftfreq(len(sig), 1.0 / fs)[:len(sig) // 2 + 1])
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
    sf = scipy.stats.mstats.gmean(magnitude_spectrum) / np.mean(magnitude_spectrum)
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
    sr = [np.where(np.cumsum(power_spectrum[t, :]) >= k * spectral_sum[t])[0][0]
          for t in range(tbins - 1)]
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
    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2, axis=1)) / fbins

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

    for bin in spectrum:
        # Compute center frequency
        f = ((fs / 2.0) / len(spectrum)) * bin_count
        numerator = numerator + (((f - centroid)**2) * abs(bin))
        denominator = denominator + abs(bin)
        bin_count = bin_count + 1

    return np.sqrt((numerator * 1.0) / denominator)


def zero_crossing_rate(sig, fs, block_length=256):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(sig) / block_length))

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(fs)))
    zcr = []

    for i in range(0,num_blocks-1):
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(sig)])

        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(sig[start:stop]))))
        zcr.append(zc)

    return np.asarray(zcr), np.asarray(timestamps)

def root_mean_square(sig, fs, block_length=256):
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(sig)/block_length))

    # when do these blocks begin (time in seconds)?
    tstamps = (np.arange(0, num_blocks - 1) * (block_length / float(fs)))

    rms = []

    for i in range(0,num_blocks-1):

        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(sig)])

        rms_seg = np.sqrt(np.mean(sig[start:stop]**2))
        rms.append(rms_seg)

    return np.asarray(rms), np.asarray(tstamps)

def spectral_bandwidth(sig, fs):
    return []

def extract_feats(sig, fs):
    """
    Compute the spectral features.

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    feats       = {}
    spectrum    = np.abs(np.fft.rfft(sig))
    frequencies = np.fft.rfftfreq(len(sig), d=1. / fs)
    amplitudes  = spectrum / spectrum.sum()

    import matplotlib.pyplot as plt
    plt.stem(frequencies[:8000:25], spectrum[:8000:25], markerfmt=' ')
    plt.show()

    # stats
    mean_frequency     = (frequencies * amplitudes).sum()
    peak_frequency     = frequencies[np.argmax(amplitudes)]
    frequencies_std    = frequencies.std()
    amplitudes_cum_sum = np.cumsum(amplitudes)
    mode_frequency     = frequencies[amplitudes.argmax()]
    median_frequency   = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.50]) + 1]
    frequencies_q25    = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.25]) + 1]
    frequencies_q75    = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.75]) + 1]

    # general stats
    feats["duration"] = len(sig) / float(fs)
    feats["spectrum"] = spectrum

    # assign spectral stats
    feats["meanfreq"]           = frequencies.mean()
    feats["sd"]                 = frequencies.std()
    feats["medianfreq"]         = np.median(frequencies)
    feats["q25"]                = frequencies_q25
    feats["q75"]                = frequencies_q75
    feats["iqr"]                = feats["q75"] - feats["q25"]
    feats["freqs_skewness"]     = scipy.stats.skew(frequencies)
    feats["freqs_kurtosis"]     = scipy.stats.kurtosis(frequencies)
    feats["spectral_entropy"]   = scipy.stats.entropy(amplitudes)
    feats["spectral_flatness"]  = spectral_flatness(sig)
    feats["modef"]              = mode_frequency
    feats["peakf"]              = frequencies[np.argmax(amplitudes)]
    feats["spectral_centroid"]  = spectral_centroid(sig, fs)
    feats["spectral_bandwidth"] = spectral_bandwidth(sig, fs)
    feats["spectral_spread"]    = spectral_spread(feats["spectral_centroid"], feats["spectrum"], fs)
    feats["spectral_flatness"]  = spectral_flatness(sig)
    feats["spectral_rolloff"]   = spectral_rolloff(sig, fs)

    # compute energy
    frame_hop = 256
    frame_len = 512
    feats["energy"] = np.array([np.sum(abs(x[i: i + frame_len]**2))
                                for i in range(len(sig), frame_hop)])

    # compute root-mean-square (RMS).
    feats["rms"] = root_mean_square(sig=sig, fs=fs)

    # compute the zero-crossing rate of an audio time series
    feats["zcr"] = zero_crossing_rate(sig=sig, fs=fs)

    # spectral stats
    feats["spectral_mean"]     = np.mean(spectrum)
    feats["spectral_rms"]      = np.sqrt(np.mean(spectrum**2))
    feats["spectral_std"]      = np.std(spectrum)
    feats["spectral_variance"] = np.var(spectrum)

    # assign fundamental frequencies stats
    fund_freqs = compute_fund_freqs(sig=sig, fs=fs)
    feats["meanfun"] = fund_freqs.mean()
    feats["minfun"] = fund_freqs.min()
    feats["maxfun"] = fund_freqs.max()

    # assign dominant frequencies stats
    dom_freqs, mod_idx = compute_dom_freqs_and_mod_index(sig=sig, fs=fs)
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
