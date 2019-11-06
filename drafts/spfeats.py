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
from .. import features
from ..utils import preprocessing as proc
from ..frequencies.fundamental_frequencies import DominantFrequenciesExtractor
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
        modulation_index = changes.mean() / dfrange
    return dom_freqs, mod_index


def compute_fund_freqs_feats(sig, fs):
    feats = {}
    # assign fundamental frequencies stats
    fund_freqs = compute_fund_freqs()
    feats["meanfun"] = fund_freqs.mean()
    feats["minfun"] = fund_freqs.min()
    feats["maxfun"] = fund_freqs.max()
    return fundfreqs


def compute_dom_freqs_feats(sig, fs):
    feats = {}
    # assign dominant frequencies stats
    dom_freqs, mod_idx = compute_dom_freqs_and_mod_index(
        sig=sig, fs=fs)
    feats["meandom"] = dom_freqs.mean()
    feats["mindom"] = dom_freqs.min()
    feats["maxdom"] = dom_freqs.max()

    # range of dominant frequency measured across acoustic signal
    feats["dfrange"] = feats["maxdom"] - feats["mindom"]

    # modulation index: Calculated as the accumulated absolute difference
    # between adjacent measurements of fundamental frequencies divided by the
    # frequency range
    feats["modindex"] = mod_idx
    return dom_freqs


def compute_spectral_spread(centroid, spectrum, fs):
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


def extract_feats(sig, fs):
    """
    Compute the spectral spread (basically a variance of the spectrum around the
    spectral centroid)

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    feats = {}
    spectrum = np.abs(np.fft.rfft(sig))
    frequencies = np.fft.rfftfreq(len(sig), d=1. / fs)
    amplitudes = spectrum / spectrum.sum()

    import matplotlib.pyplot as plt
    plt.stem(frequencies[:8000:25], spectrum[:8000:25], markerfmt=' ')
    plt.show()

    # stats
    mean_frequency = (frequencies * amplitudes).sum()
    peak_frequency = frequencies[np.argmax(amplitudes)]
    frequencies_std = frequencies.std()
    amplitudes_cum_sum = np.cumsum(amplitudes)
    median_frequency = frequencies[
        len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.5]) + 1]
    mode_frequency = frequencies[amplitudes.argmax()]
    frequencies_q25 = frequencies[
        len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.25]) + 1]
    frequencies_q75 = frequencies[
        len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.75]) + 1]

    # general stats
    feats["duration"] = len(self.signal) / float(self.rate)
    feats["spectrum"] = spectrum

    # assign spectral stats
    feats["meanfreq"] = frequencies.mean()
    feats["sd"] = frequencies.std()
    feats["medianfreq"] = np.median(frequencies)
    feats["q25"] = frequencies_q25
    feats["q75"] = frequencies_q75
    feats["iqr"] = feats["q75"] - feats["q25"]
    feats["freqs_skewness"] = scipy.stats.skew(frequencies)
    feats["freqs_kurtosis"] = scipy.stats.kurtosis(frequencies)
    feats["spectral_entropy"] = scipy.stats.entropy(amplitudes)
    feats["spectral_flatness"] = librosa.feature.spectral_flatness(signal)
    feats["modef"] = mode_frequency
    feats["peakf"] = frequencies[np.argmax(amplitudes)]
    feats["spectral_centroid"] = librosa.feature.spectral_centroid(
        signal, rate)
    feats["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
        signal, rate)
    feats["spectral_spread"] = self.compute_spectral_spread(
        feats["spectral_centroid"], feats["spectrum"])
    feats["spectral_flatness"] = librosa.feature.spectral_flatness(signal)
    feats["spectral_rolloff"] = librosa.feature.spectral_rolloff(signal, rate)
    feats["poly_features"] = librosa.feature.poly_features(signal, order=2)
    feats["tonnetz"] = librosa.feature.tonnetz(signal, rate)


def compute_temporal_feats(signal, rate):
    # temporal features
    feats["cqt"] = librosa.core.cqt(signal, rate)  # Constant Q Transform
    feats["dct"] = librosa.feature.spectral_centroid(
        signal, rate)  # Discrete Cosine Transform
    feats["energy"] = librosa.feature.spectral_centroid(signal, rate)  # Energy
    # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    feats["rms"] = librosa.feature.rms(signal)
    # Compute the zero-crossing rate of an audio time series
    feats["zcr"] = librosa.feature.zero_crossing_rate(signal)
    return feats


def spectral_feats(sig):
    fourrier = proc.fft(sig)
    spectrum = np.abs(fourrier)
    feats = {}
    # spectral stats
    feats["spectral_mean"] = np.mean(spectrum)
    feats["spectral_rms"] = np.sqrt(np.mean(spectrum**2))
    feats["spectral_std"] = np.std(spectrum)
    feats["spectral_variance"] = np.var(spectrum)
    feats["spectral_skewness"] = scipy.stats.skew(spectrum)
    feats["spectral_kurtosis"] = scipy.stats.kurtosis(spectrum)
    feats["spectral_entropy"] = scipy.stats.entropy(spectrum)
    feats["energy"] = np.sum(np.abs(spectrum**2))
    feats["centroid"] = np.sum(fourrier * spectrum) / np.sum(fourrier)
    feats["rolloff"] = librosa.feature.spectral_rolloff(signal, rate)
    feats["spread"] = spectral_feats(sig, rate)
    return feats
