"""
Compute the following spectral stats:    http://ijeee.iust.ac.ir/article-1-1074-en.pdf
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
"""
import scipy 
import numpy as np
import spafe.features
import spafe.frequencies
import spafe.utils.processing as proc


def compute_fundamental_frequencies(sig, fs):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # fundamental frequencies calculations
    fundamental_frequencies_extractor       = spafe.frequencies.FundamentalFrequenciesExtractor(False)
    pitches, harmonic_rates, argmins, times = fundamental_frequencies_extractor.main(fs, sig)
    return pitches

def compute_dominant_frequencies_and_modulation_index(sig, fs):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    # dominant frequencies calculations
    dominant_frequencies_extractor = spafe.frequencies.DominantFrequenciesExtractor()
    dominant_frequencies           = dominant_frequencies_extractor.main(sig, fs)

    # modulation index calculation
    changes = np.abs(dominant_frequencies[:-1] - dominant_frequencies[1:])
    dfrange = dominant_frequencies.max() - dominant_frequencies.min()
    if dominant_frequencies.min() == dominant_frequencies.max():
        modulation_index = 0
    else:
        modulation_index = changes.mean() / dfrange
    return dominant_frequencies, modulation_index

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
        f           = ((fs / 2.0) / len(spectrum)) * bin_count
        numerator   = numerator + (((f - centroid) ** 2) * abs(bin))
        denominator = denominator + abs(bin)
        bin_count   = bin_count + 1

    return np.sqrt((numerator * 1.0) / denominator)

def extract_feats(sig, fs):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)

    Args:
        centroid (float) : spectral centroid.
        spectrum (array) : spectrum array.

    Returns:
        (float) spectral spread.
    """
    feats           = {}
    spectrum             = np.abs(np.fft.rfft(sig))
    frequencies          = np.fft.rfftfreq(len(sig), d=1. / fs)
    amplitudes           = spectrum / spectrum.sum()

    import matplotlib.pyplot as plt
    plt.stem(frequencies[:8000:25], spectrum[:8000:25], markerfmt=' ')
    plt.show()

    # stats
    mean_frequency     = (frequencies * amplitudes).sum()
    peak_frequency     = frequencies[np.argmax(amplitudes)]
    frequencies_std    = frequencies.std()
    amplitudes_cum_sum = np.cumsum(amplitudes)
    median_frequency   = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.5]) + 1]
    mode_frequency     = frequencies[amplitudes.argmax()]
    frequencies_q25    = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.25]) + 1]
    frequencies_q75    = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.75]) + 1]

    # general stats
    feats["duration"] = len(self.signal) / float(self.rate)
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
    feats["sfm"]                = librosa.feature.spectral_flatness(signal)
    feats["modef"]              = mode_frequency
    feats["peakf"]              = frequencies[np.argmax(amplitudes)]
    feats["spectral_centroid"]  = librosa.feature.spectral_centroid(signal,  rate)
    feats["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(signal, rate)
    feats["spectral_spread"]    = self.compute_spectral_spread(feats["spectral_centroid"], feats["spectrum"])
    feats["spectral_flatness"]  = librosa.feature.spectral_flatness(signal)
    feats["spectral_rolloff"]   = librosa.feature.spectral_rolloff(signal, rate)
    feats["poly_features"]      = librosa.feature.poly_features(signal, order=2)
    feats["tonnetz"]            = librosa.feature.tonnetz(signal, rate)

def compute_fundfreqs_feats(sig, fs):
    feats = {}
    # assign fundamental frequencies stats
    fundfreqs        = compute_fundamental_frequencies()
    feats["meanfun"] = fundfreqs.mean()  # average of fundamental frequency measured across acoustic signal
    feats["minfun"]  = fundfreqs.min()   # minimum fundamental frequency measured across acoustic signal
    feats["maxfun"]  = fundfreqs.max()   # maximum fundamental frequency measured across acoustic signal
    return fundfreqs

def compute_domfreqs_feats(sig, fs):
    feats = {}
    # assign dominant frequencies stats
    domfreqs, mod_idx = compute_dominant_frequencies_and_modulation_index(sig, fs)
    feats["meandom"]  = domfreqs.mean()                   # average of dominant frequency measured across acoustic signal
    feats["mindom"]   = domfreqs.min()                    # minimum of dominant frequency measured across acoustic signal
    feats["maxdom"]   = domfreqs.max()                    # maximum of dominant frequency measured across acoustic signal
    feats["dfrange"]  = feats["maxdom"] - feats["mindom"] # range of dominant frequency measured across acoustic signal
    feats["modindex"] = mod_idx                           # modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
    return domfreqs

def compute_temporal_feats(signal, rate):
    # temporal features
    feats["cqt"]      = librosa.core.cqt(signal, rate)                  # Constant Q Transform
    feats["dct"]      = librosa.feature.spectral_centroid(signal, rate) # Discrete Cosine Transform
    feats["energy"]   = librosa.feature.spectral_centroid(signal, rate) # Energy
    feats["rms"]      = librosa.feature.rms(signal)                     # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    feats["zcr"]      = librosa.feature.zero_crossing_rate(signal)      # Compute the zero-crossing rate of an audio time series
    return feats

def spectral_feats(sig):
    fourrier = proc.fft(sig)
    spectrum = np.abs(fourrier)
    feats    = {}
    # spectral stats
    feats["spectral_mean"]     = np.mean(spectrum) 
    feats["spectral_rms"]      = np.sqrt(np.mean(spectrum**2))
    feats["spectral_std"]      = np.std(spectrum)
    feats["spectral_variance"] = np.var(spectrum)
    feats["spectral_skewness"] = scipy.stats.skew(spectrum)
    feats["spectral_kurtosis"] = scipy.stats.kurtosis(spectrum)
    feats["spectral_entropy"]  = scipy.stats.entropy(spectrum)
    feats["energy"]            = np.sum(np.abs(spectrum**2)) 
    feats["centroid"]          = np.sum(fourrier * spectrum) / np.sum(fourrier)
    feats["rolloff"]           = librosa.feature.spectral_rolloff(signal,  rate)
    feats["spread"]            = spectral_feats(sig, rate)
    return feats


