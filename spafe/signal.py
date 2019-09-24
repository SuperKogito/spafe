import numpy
import scipy
import librosa
import spafe.features
import spafe.frequencies


class Signal:
    """
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
    def __init__(self, file_name):
        self.file_name          = file_name
        self.signal, self.rate  = librosa.load(self.file_name)
        self.signal, self.rate  = self.signal[:8000:], self.rate
        self.features_extractor = spafe.features.FeaturesExtractor(self.signal, self.rate)
        self.properties         = self.extract_properties()

        # general stats
        self.duration = self.properties["duration"]

        # mfccs feats
        self.mfcc     = self.properties["mfcc"]
        self.mfccd1   = self.properties["mfcc-deltas-1"]
        self.mfccd2   = self.properties["mfcc-deltas-2"]

        # linear feats
        self.lpc      = self.properties["lpc"]      # LPC, with order = len(self)-1
        self.lpcc     = self.properties["lpcc"]     # LPCC, with order = len(self)-1
        self.lsp      = self.properties["lsp"]      # LSP/LSF, with order = len(fixedFrames[0])-1

        # spectral stats
        self.meanfreq           = self.properties["meanfreq"]
        self.sd                 = self.properties["sd"]
        self.medianfreq         = self.properties["medianfreq"]
        self.q25                = self.properties["q25"]
        self.q75                = self.properties["q75"]
        self.iqr                = self.properties["iqr"]
        self.modef              = self.properties["modef"]
        self.peakf              = self.properties["peakf"]
        self.meanfun            = self.properties["meanfun"]
        self.minfun             = self.properties["minfun"]
        self.maxfun             = self.properties["maxfun"]
        self.meandom            = self.properties["meandom"]
        self.mindom             = self.properties["mindom"]
        self.maxdom             = self.properties["maxdom"]
        self.dfrange            = self.properties["dfrange"]
        self.modindx            = self.properties["modindex"]
        self.freqs_kutosis      = self.properties["freqs_kurtosis"]
        self.spectral_entropy   = self.properties["spectral_entropy"]
        self.spectral_flatness  = self.properties["spectral_flatness"]
        self.spectral_centroid  = self.properties["spectral_centroid"]
        self.spectral_bandwidth = self.properties["spectral_bandwidth"]
        self.spectral_rolloff   = self.properties["spectral_rolloff"]
        self.spectral_spread    = self.properties["spectral_spread"]
        self.spectral_variance  = self.properties["spectral_variance"]

        # temporal features
        self.cqt      = self.properties["cqt"]            # Constant Q Transform
        self.dct      = self.properties["dct"]            # Discrete Cosine Transform
        self.energy   = self.properties["energy"]         # Energy
        self.rms      = self.properties["rms"]            # Root-mean-squared amplitude
        self.spectrum = self.properties["spectrum"]       # spectrum fft
        self.zcr      = self.properties["zcr"]            # Zero-crossing raate

        # other features
        self.chroma   = self.properties["chroma"]             # Chroma vector
        self.crest    = self.properties["crest"]              # Spectral Crest Factor
        self.idct     = self.properties["idct"]               # Inverse DCT
        self.ifft     = self.properties["ifft"]               # Inverse FFT

    def compute_fundamental_frequencies(self):
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
        pitches, harmonic_rates, argmins, times = fundamental_frequencies_extractor.main(self.rate, self.signal)
        return pitches

    def compute_dominant_frequencies_and_modulation_index(self):
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
        dominant_frequencies           = dominant_frequencies_extractor.main(self.signal, self.rate)

        # modulation index calculation
        changes = numpy.abs(dominant_frequencies[:-1] - dominant_frequencies[1:])
        dfrange = dominant_frequencies.max() - dominant_frequencies.min()
        if dominant_frequencies.min() == dominant_frequencies.max():
            modulation_index = 0
        else:
            modulation_index = changes.mean() / dfrange
        return dominant_frequencies, modulation_index

    def extract_properties(self):
        """
        Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)

        Args:
            centroid (float) : spectral centroid.
            spectrum (array) : spectrum array.

        Returns:
            (float) spectral spread.
        """
        properties           = {}
        signal, rate         = self.signal, self.rate
        spectrum             = numpy.abs(numpy.fft.rfft(signal))
        frequencies          = numpy.fft.rfftfreq(len(signal), d=1. / rate)
        amplitudes           = spectrum / spectrum.sum()

        import matplotlib.pyplot as plt
        plt.stem(frequencies[:8000:25], spectrum[:8000:25], markerfmt=' ')
        plt.show()

        # stats
        mean_frequency       = (frequencies * amplitudes).sum()
        peak_frequency       = frequencies[numpy.argmax(amplitudes)]
        frequencies_std      = frequencies.std()
        amplitudes_cum_sum   = numpy.cumsum(amplitudes)
        median_frequency     = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.5]) + 1]
        mode_frequency       = frequencies[amplitudes.argmax()]
        frequencies_q25      = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.25]) + 1]
        frequencies_q75     = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.75]) + 1]

        # general stats
        properties["duration"] = len(self.signal) / float(self.rate)
        properties["spectrum"] = spectrum

        # assign spectral stats
        properties["meanfreq"]           = frequencies.mean()
        properties["sd"]                 = frequencies.std()
        properties["medianfreq"]         = numpy.median(frequencies)
        properties["q25"]                = frequencies_q25
        properties["q75"]                = frequencies_q75
        properties["iqr"]                = properties["q75"] - properties["q25"]
        properties["freqs_skewness"]     = scipy.stats.skew(frequencies)
        properties["freqs_kurtosis"]     = scipy.stats.kurtosis(frequencies)
        properties["spectral_entropy"]   = scipy.stats.entropy(amplitudes)
        properties["sfm"]                = librosa.feature.spectral_flatness(signal)
        properties["modef"]              = mode_frequency
        properties["peakf"]              = frequencies[numpy.argmax(amplitudes)]
        properties["spectral_centroid"]  = librosa.feature.spectral_centroid(signal,  rate)
        properties["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(signal, rate)
        properties["spectral_spread"]    = self.compute_spectral_spread(properties["spectral_centroid"], properties["spectrum"])
        properties["spectral_flatness"]  = librosa.feature.spectral_flatness(signal)
        properties["spectral_rolloff"]   = librosa.feature.spectral_rolloff(signal, rate)
        properties["poly_features"]      = librosa.feature.poly_features(signal, order=2)
        properties["tonnetz"]            = librosa.feature.tonnetz(signal, rate)

        # assign fundamental frequencies stats
        fundamental_frequencies = self.compute_fundamental_frequencies()
        properties["meanfun"]   = fundamental_frequencies.mean()      # average of fundamental frequency measured across acoustic signal
        properties["minfun"]    = fundamental_frequencies.min()       # minimum fundamental frequency measured across acoustic signal
        properties["maxfun"]    = fundamental_frequencies.max()       # maximum fundamental frequency measured across acoustic signal

        # assign dominant frequencies stats
        dominant_frequencies, modulation_index = self.compute_dominant_frequencies_and_modulation_index()
        properties["meandom"]  = dominant_frequencies.mean()                   # average of dominant frequency measured across acoustic signal
        properties["mindom"]   = dominant_frequencies.min()                    # minimum of dominant frequency measured across acoustic signal
        properties["maxdom"]   = dominant_frequencies.max()                    # maximum of dominant frequency measured across acoustic signal
        properties["dfrange"]  = properties["maxdom"] - properties["mindom"]   # range of dominant frequency measured across acoustic signal
        properties["modindex"] = modulation_index                              # modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range

        # assign features
        properties["chroma"]   = librosa.feature.chroma_stft(signal, rate)       # Chroma vector
        properties["crest"]    = []             # Spectral Crest Factor
        properties["idct"]     = []             # Inverse DCT
        properties["ifft"]     = []             # Inverse FFT

        # assign MFCC features
        mfcc_features               = self.features_extractor.get_mfcc_features(signal, rate)
        properties["mfcc"]          = mfcc_features[0]
        properties["mfcc-deltas-1"] = mfcc_features[1]
        properties["mfcc-deltas-2"] = mfcc_features[2]

        # assign linear features
        linear_features    = self.features_extractor.get_linear_features(signal, rate)
        properties["lpc"]  = linear_features[0]
        properties["lpcc"] = linear_features[1]
        properties["lsp"]  = linear_features[2]

        # features
        properties["chroma_stft"] = librosa.feature.chroma_stft(signal, rate)  # Compute a chromagram from a waveform or power spectrogram.
        properties["chroma_cqt"]  = librosa.feature.chroma_cqt(signal,  rate)  # Constant-Q chromagram
        properties["chroma_cens"] = librosa.feature.chroma_cens(signal, rate)  # Computes the chroma variant “Chroma Energy Normalized” (CENS), following [R674badebce0d-1].

        # temporal features
        properties["cqt"]      = librosa.core.cqt(signal, rate)                  # Constant Q Transform
        properties["dct"]      = librosa.feature.spectral_centroid(signal, rate) # Discrete Cosine Transform
        properties["energy"]   = librosa.feature.spectral_centroid(signal, rate) # Energy
        properties["rms"]      = librosa.feature.rms(signal)                     # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
        properties["zcr"]      = librosa.feature.zero_crossing_rate(signal)      # Compute the zero-crossing rate of an audio time series

        # spectral stats
        properties["centroid"] = librosa.feature.spectral_centroid(signal, rate)
        properties["rolloff"]  = librosa.feature.spectral_rolloff(signal,  rate)
        properties["skewness"] = librosa.feature.spectral_centroid(signal, rate)
        properties["spread"]   = librosa.feature.spectral_centroid(signal, rate)
        properties["spectral_variance"] = numpy.var(abs(spectrum))

        return properties



    def compute_spectral_spread(self, centroid, spectrum):
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
            f           = ((self.rate / 2.0) / len(spectrum)) * bin_count
            numerator   = numerator + (((f - centroid) ** 2) * abs(bin))
            denominator = denominator + abs(bin)
            bin_count   = bin_count + 1

        return numpy.sqrt((numerator * 1.0) / denominator)


if __name__ == "__main__":
    signal = Signal("test.wav")
    print(signal.properties.keys())
