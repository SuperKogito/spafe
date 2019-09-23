import numpy
import scipy
import librosa
import itertools
from FundFreqs import FundamentalFrequencies
from DomFreqs import DominantFrequenciesExtractor   


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
        self.file_name         = file_name 
        self.signal, self.rate = librosa.load(self.file_name)
        self.signal, self.rate = self.signal[:8000:], self.rate
        self.properties        = self.extract_properties()
        
        # general stats
        self.duration = self.properties["duration"] = len(self.signal) / float(self.rate)
        
        # mfccs feats
        self.mfcc     = self.properties["mfcc"]          
        self.mfccd1   = self.properties["mfcc-deltas-1"] 
        self.mfccd2   = self.properties["mfcc-deltas-2"]
        
        # linear feats
        self.lpc      = self.properties["lpc"]      # LPC, with order = len(self)-1
        self.lpcc     = self.properties["lpcc"]     # LPCC, with order = len(self)-1
        self.lsp      = self.properties["lsp"]      # LSP/LSF, with order = len(fixedFrames[0])-1
        
        # spectral stats
        self.meanfreq   = self.properties["meanfreq"]       # mean frequency (in kHz)
        self.sd         = self.properties["sd"]             # standard deviation of frequency
        self.medianfreq = self.properties["medianfreq"]     # median frequency (in kHz)
        self.q25        = self.properties["q25"]            # first quantile (in kHz)
        self.q75        = self.properties["q75"]            # third quantile (in kHz)
        self.iqr        = self.properties["iqr"]            # interquantile range (in kHz)
        self.mode       = self.properties["mode"]           # mode frequency
        self.peakf      = self.properties["peakf"]          # peak frequency (frequency with highest energy)
        self.meanfun    = self.properties["meanfun"]        # average of fundamental frequency measured across acoustic signal
        self.minfun     = self.properties["minfun"]         # minimum fundamental frequency measured across acoustic signal
        self.maxfun     = self.properties["maxfun"]         # maximum fundamental frequency measured across acoustic signal
        self.meandom    = self.properties["meandom"]        # average of dominant frequency measured across acoustic signal
        self.mindom     = self.properties["mindom"]         # minimum of dominant frequency measured across acoustic signal
        self.maxdom     = self.properties["maxdom"]         # maximum of dominant frequency measured across acoustic signal
        self.dfrange    = self.properties["dfrange"]        # range of dominant frequency measured across acoustic signal
        self.modindx    = self.properties["modindex"]       # modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
        self.freqs_skewness     = self.properties["freqs_skewness"]    
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
        # fundamental frequencies calculations   
        fundamental_frequencies_extractor = FundamentalFrequencies(False)
        pitches, harmonic_rates, argmins, times = fundamental_frequencies_extractor.main(self.rate, self.signal)
        return pitches

    def compute_dominant_frequencies_and_modulation_index(self):
        # dominant frequencies calculations   
        dominant_frequencies_extractor = DominantFrequenciesExtractor()
        dominant_frequencies           = dominant_frequencies_extractor.main("sample2.wav")
         
        # modulation index calculation  
        changes = numpy.abs(dominant_frequencies[:-1] - dominant_frequencies[1:])
        dfrange = dominant_frequencies.max() - dominant_frequencies.min()
        if dominant_frequencies.min() == dominant_frequencies.max():
            modulation_index = 0
        else: 
            modulation_index = changes.mean() / dfrange
        return dominant_frequencies, modulation_index

    def extract_properties(self): 
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
        frequencies_q75      = frequencies[len(amplitudes_cum_sum[amplitudes_cum_sum <= 0.75]) + 1]
        
        # assign spectral stats
        properties["meanfreq"]   = frequencies.mean()         # mean frequency (in kHz)
        properties["sd"]         = frequencies.std()          # standard deviation of frequency
        properties["medianfreq"] = numpy.median(frequencies)  # median frequency (in kHz)
        properties["q25"]        = frequencies_q25            # first quantile (in kHz)
        properties["q75"]        = frequencies_q75            # third quantile (in kHz)
        properties["iqr"]        = properties["q75"] - properties["q25"]  # interquantile range (in kHz)
        properties["freqs_skewness"]   = scipy.stats.skew(frequencies)          # skewness (see note in specprop description)
        properties["freqs_kurtosis"]   = scipy.stats.kurtosis(frequencies)      # kurtosis (see note in specprop description)
        properties["spectral_entropy"] = scipy.stats.entropy(amplitudes)        # spectral entropy
        properties["sfm"]   = librosa.feature.spectral_flatness(signal)         # spectral flatness
        properties["mode"]  = mode_frequency                                    # mode frequency
        properties["peakf"] = frequencies[numpy.argmax(amplitudes)]             # peak frequency (frequency with highest energy)
        
        properties["spectral_centroid"]  = librosa.feature.spectral_centroid(signal,  rate)              # Compute the spectral centroid.
        properties["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(signal, rate)              # Compute p’th-order spectral bandwidth.
        properties["spectral_spread"]    = self.compute_spectral_spread(properties["spectral_centroid"])                     # Compute spectral spread        
        properties["spectral_flatness"]  = librosa.feature.spectral_flatness(signal)                     # Compute spectral flatness
        properties["spectral_rolloff"]   = librosa.feature.spectral_rolloff(signal, rate)                # Compute roll-off frequency.
        properties["poly_features"]      = librosa.feature.poly_features(signal, order=2)                # Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
        properties["tonnetz"]            = librosa.feature.tonnetz(signal, rate) 	                     # Computes the tonal centroid features (tonnetz), following the method of [Recf246e5a035-1].
    
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
        mfcc_features               = self.get_mfcc_features(signal, rate)
        properties["mfcc"]          = mfcc_features[0]
        properties["mfcc-deltas-1"] = mfcc_features[1]
        properties["mfcc-deltas-2"] = mfcc_features[2]               # MFCC (vectorized implementation)

        # assign linear features 
        linear_features    = self.get_linear_features(signal, rate)    
        properties["lpc"]  = linear_features[0]                  # LPC, with order = len(self)-1
        properties["lpcc"] = linear_features[1]
        properties["lsp"]  = linear_features[2]                  # LSP/LSF, with order = len(fixedFrames[0])-1

        # features 
        properties["chroma_stft"] = librosa.feature.chroma_stft(signal, rate)                            # Compute a chromagram from a waveform or power spectrogram.
        properties["chroma_cqt"]  = librosa.feature.chroma_cqt(signal,  rate)                            # Constant-Q chromagram
        properties["chroma_cens"] = librosa.feature.chroma_cens(signal, rate)                            # Computes the chroma variant “Chroma Energy Normalized” (CENS), following [R674badebce0d-1].

        # temporal features
        properties["cqt"]      = librosa.core.cqt(signal, rate)                  # Constant Q Transform       
        properties["dct"]      = librosa.feature.spectral_centroid(signal, rate) # Discrete Cosine Transform
        properties["energy"]   = librosa.feature.spectral_centroid(signal, rate) # Energy
        properties["rms"]      = librosa.feature.rms(signal)                     # Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
        properties["spectrum"] = spectrum                                        # fft spectrum
        properties["zcr"]      = librosa.feature.zero_crossing_rate(signal)      # Compute the zero-crossing rate of an audio time series 

        # spectral stats 
        properties["centroid"] = librosa.feature.spectral_centroid(signal, rate)             # Plot using matplotlib
        properties["rolloff"]  = librosa.feature.spectral_rolloff(signal, rate)                # Spectral Rolloff
        properties["skewness"] = librosa.feature.spectral_centroid(signal, rate)                # Spectral Skewness
        properties["spread"]   = librosa.feature.spectral_centroid(signal, rate)                  # Spectral Spread
        properties["spectral_variance"] = numpy.var(abs(self.spectrum))                # Spectral Variance

        properties["duration"] = librosa.feature.spectral_centroid(signal, rate)                # length of signal
        return properties



    def compute_spectral_spread(self, centroid):
        """
        Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)
        """
        binNumber   = 0
        numerator   = 0
        denominator = 0
        
        for bin in self.spectrum:
            # Compute center frequency
            f = (self.rate / 2.0) / len(self.spectrum)
            f = f * binNumber
        
            numerator   = numerator + (((f - centroid) ** 2) * abs(bin))
            denominator = denominator + abs(bin)
            binNumber   = binNumber + 1

        return numpy.sqrt((numerator * 1.0) / denominator)
           
    def get_mfcc_features(self, signal, rate):
        # Extract MFCCs, MFCC deltas, MFCC dosuble deltas
        mfcc        = librosa.feature.mfcc(signal, n_mfcc=13)
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return [mfcc, delta_mfcc, delta2_mfcc]
    
    def get_linear_features(self, signal, rate):
        # Extract LPCs, LPCCs and LSPs
        lpc  = librosa.core.lpc(signal, order=3) 
        lpcc = self.lpcc(lpc)
        lsp  = self.lsp(lpc)
        return [lpc, lpcc, lsp]

    def lpcc(self, seq, order=None):
        '''
        Function: lpcc
        Summary: Computes the linear predictive cepstral compoents. Note: Returned values are in the frequency domain
        Examples: audiofile = AudioFile.open('file.wav',16000)
                  frames = audiofile.frames(512,np.hamming)
                  for frame in frames:
                    frame.lpcc()
                  Note that we already preprocess in the Frame class the lpc conversion!
        Attributes:
            @param (seq):A sequence of lpc components. Need to be preprocessed by lpc()
            @param (err_term):Error term for lpc sequence. Returned by lpc()[1]
            @param (order) default=None: Return size of the array. Function returns order+1 length array. Default is len(seq)
        Returns: List with lpcc components with default length len(seq), otherwise length order +1
        '''
        if order is None: order = len(seq) - 1
        lpcc_coeffs = [-seq[0], -seq[1]]
        for n in range(2, order + 1):
            print(n, len(seq), order, seq[n-1-1], lpcc_coeffs[1])
            # Use order + 1 as upper bound for the last iteration
            upbound    = (order + 1 if n > order else n)
            lpcc_coef  = -sum(i * lpcc_coeffs[i] * seq[n - i - 1] for i in range(1, upbound)) * 1. / upbound
            lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
            lpcc_coeffs.append(lpcc_coef)
        return lpcc_coeffs

    def lsp(self, lpcseq, rectify=True):
        '''
        Function: lsp
        Summary: Computes Line spectrum pairs ( also called  line spectral frequencies [lsf]). Does not use any fancy algorithm except np.roots to solve
        for the zeros of the given polynom A(z) = 0.5(P(z) + Q(z))
        Examples: audiofile = AudioFile.open('file.wav',16000)
                  frames = audiofile.frames(512,np.hamming)
                  for frame in frames:
                    frame.lpcc()
        Attributes:
            @param (lpcseq):The sequence of lpc coefficients as \sum_k=1^{p} a_k z^{-k}
            @param (rectify) default=True: If true returns only the values >= 0, since the result is symmetric. If all values are wished, specify rectify = False
        Returns: A list with same length as lpcseq (if rectify = True), otherwise 2*len(lpcseq), which represents the line spectrum pairs
        '''
        import numpy as np
        # We obtain 1 - A(z) +/- z^-(p+1) * (1 - A(z))
        # After multiplying everything through it becomes
        # 1 - \sum_k=1^p a_k z^{-k} +/- z^-(p+1) - \sum_k=1^p a_k z^{k-(p+1)}
        # Thus we have on the LHS the usual lpc polynomial and on the RHS we need to reverse the coefficient order
        # We assume further that the lpc polynomial is a valid one ( first coefficient is 1! )
    
        # the rhs does not have a constant expression and we reverse the coefficients
        rhs = [0] + lpcseq[::-1] + [1]
        # init the P and the Q polynomials
        P, Q = [], []
        # Assuming constant coefficient is 1, which is required. Moreover z^{-p+1} does not exist on the lhs, thus appending 0
        lpcseq = [1] + lpcseq[:] + [0]
        for l,r in itertools.zip_longest(lpcseq, rhs):
            P.append(l + r)
            Q.append(l - r)
        # Find the roots of the polynomials P,Q ( numpy assumes we have the form of: p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
        # mso we need to reverse the order)
        p_roots = np.roots(P[::-1])
        q_roots = np.roots(Q[::-1])
        # Keep the roots in order
        lsf_p = sorted(np.angle(p_roots))
        lsf_q = sorted(np.angle(q_roots))
        # print sorted(lsf_p+lsf_q),len([i for  i in lsf_p+lsf_q if i > 0.])
        if rectify:
            # We only return the positive elements, and also remove the final Pi (3.14) value at the end,
            # since it always occurs
            return sorted(i for i in lsf_q + lsf_p if (i > 0))[:-1]
        else:
            # Remove the -Pi and +pi at the beginning and end in the list
            return sorted(i for i in lsf_q + lsf_p)[1:-1]
        
        
if __name__ == "__main__":
    signal = Signal("sample2.wav")
    print(signal.properties.keys())