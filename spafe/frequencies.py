"""
Credits to: 
    Patrice Guyot. (2018, April 19). 
    Fast Python implementation of the Yin algorithm (Version v1.1.1). 
    Zenodo. http://doi.org/10.5281/zenodo.1220947
"""
import time
import scipy
import numpy
import librosa
import matplotlib.pyplot as plt


class FundamentalFrequenciesExtractor:
    def __init__(self, debug=False):
        self.debug = debug

    def differenceFunction(self, x, N, tau_max):
        """
        Compute difference function of data x. This corresponds to equation (6) in [1]
        Fastest implementation. Use the same approach than differenceFunction_scipy.
        This solution is implemented directly with Numpy fft.
        
        Args: 	    
            x       (array) : audio data
            N       (int)   : length of data
            tau_max (int)   : integration window size
 
        Returns: 	    
            (list) : difference function
        """
        x = numpy.array(x, numpy.float64)
        w = x.size
        x_cumsum = numpy.concatenate((numpy.array([0]), (x * x).cumsum()))
        conv     = scipy.signal.fftconvolve(x, x[::-1])
        tmp      = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
        return tmp[:tau_max]


    def cumulativeMeanNormalizedDifferenceFunction(self, df, N):
        """
        Compute cumulative mean normalized difference function (CMND).
        This corresponds to equation (8) in [1].
        
        Args: 	    
            df      (list) : Difference function
            N       (int)  : length of data
            tau_max (int)  : integration window size
 
        Returns: 	    
            (list) : cumulative mean normalized difference function
        """
        cmndf = df[1:] * range(1, N) / numpy.cumsum(df[1:]).astype(float) # scipy method
        return numpy.insert(cmndf, 0, 1)
    
    def getPitch(self, cmdf, tau_min, tau_max, harmo_th=0.1):
        """
        Return fundamental period of a frame based on CMND function.
            - cmdf: Cumulative Mean Normalized Difference function
            
        Args: 	    
            tau_min  (int)   : minimum period for speech
            tau_max  (int)   : maximum period for speech
            harmo_th (float) : harmonicity threshold to determine if it is necessary to compute pitch frequency

        Returns: 	    
            (float) : fundamental period if there is values under threshold, 0 otherwise
        """
        tau = tau_min
        while tau < tau_max:
            if cmdf[tau] < harmo_th:
                while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                    tau += 1
                return tau
            tau += 1
    
        return 0    #  if unvoiced
    
    
    def compute_yin(self, sig, sr, dataFileName=None, w_len=512, w_step=256, f0_min=50, f0_max=3000, harmo_thresh=0.1):
        """
        Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
        
        Args:
            sig        (list) : Audio signal (list of float)
            sr          (int) : sampling rate (= average number of samples pro 1 second)
            w_len       (int) : size of the analysis window (in #samples)
            w_step      (int) : size of the lag between two consecutives windows (in #samples)
            f0_min      (int) : Minimum fundamental frequency that can be detected (in Hertz)
            f0_max      (int) : Maximum fundamental frequency that can be detected (in Hertz)
            harmo_tresh (int) : Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this threshold.
        
        Returns:
            (tuple) : tuple include the following
                          - pitches       : list of fundamental frequencies,
                          - harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
                          - argmins       : minimums of the Cumulative Mean Normalized DifferenceFunction
                          - times         : list of time of each estimation
        """
        if self.debug: print('Yin: compute yin algorithm')
        tau_min, tau_max = int(sr / f0_max), int(sr / f0_min)
    
        timeScale = range(0, len(sig) - w_len, w_step)  #  time values for each analysis window
        times     = [t / float(sr)    for t in timeScale]
        frames    = [sig[t:t + w_len] for t in timeScale]
    
        pitches        = [0.0] * len(timeScale)
        harmonic_rates = [0.0] * len(timeScale)
        argmins        = [0.0] * len(timeScale)
    
        for i, frame in enumerate(frames):
            # Compute YIN
            df   = self.differenceFunction(frame, w_len, tau_max)
            cmdf = self.cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
            p    = self.getPitch(cmdf, tau_min, tau_max, harmo_thresh)
    
            # Get results
            if numpy.argmin(cmdf) > tau_min:
                argmins[i] = float(sr / numpy.argmin(cmdf))
            #  A pitch was found
            if p != 0: 
                pitches[i]        = float(sr / p)
                harmonic_rates[i] = cmdf[p]
            #  No pitch, but we compute a value of the harmonic rate
            else: 
                harmonic_rates[i] = min(cmdf)
                
        return pitches, harmonic_rates, argmins, times
    
    def main(self, sr, sig, w_len=1024, w_step=256, f0_min=70, f0_max=200, harmo_thresh=0.85, audioDir="./", dataFileName=None):
        """
        Run the computation of the Yin algorithm on a example file.
        
        Args:
            sig        (list) : Audio signal (list of float)
            sr          (int) : sampling rate (= average number of samples pro 1 second)
            w_len       (int) : size of the analysis window (in #samples)
            w_step      (int) : size of the lag between two consecutives windows (in #samples)
            f0_min      (int) : Minimum fundamental frequency that can be detected (in Hertz)
            f0_max      (int) : Maximum fundamental frequency that can be detected (in Hertz)
            harmo_tresh (int) : Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this threshold.
        
        Returns:
            (tuple) : tuple include the following
                          - pitches       : list of fundamental frequencies,
                          - harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
                          - argmins       : minimums of the Cumulative Mean Normalized DifferenceFunction
                          - times         : list of time of each estimation        
        """
        start    = time.time()
        duration = len(sig) / float(sr)
        pitches, harmonic_rates, argmins, times = self.compute_yin(sig, sr, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)
        if self.debug:
            print ("Yin computed in: ", time.time() - start)
            plt.figure(figsize=(20, 10))
            plt.subplots_adjust(left   = 0.125,  right  = 0.9,   
                                bottom = 0.1,    top    = 0.9,     
                                wspace = 0.2,    hspace = 0.99)
            # plot audio data 
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
            ax1.set_title('Audio data')
            ax1.set_ylabel('Amplitude')
            
            # plot F0
            ax2 = plt.subplot(4, 1, 2)
            ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
            ax2.set_title('F0')
            ax2.set_ylabel('Frequency (Hz)')
            
            # plot Harmonic rate
            ax3 = plt.subplot(4, 1, 3, sharex=ax2)
            ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates, "-x")
            ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r', "--")
            ax3.set_title('Harmonic rate')
            ax3.set_ylabel('Rate')
            
            # plot Index of minimums of CMND
            ax4 = plt.subplot(4, 1, 4, sharex=ax2)
            ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins, "-x")
            ax4.set_title('Index of minimums of CMND')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.set_xlabel('Time (seconds)')
            plt.show()
    
        return numpy.array(pitches), harmonic_rates, argmins, times


class DominantFrequenciesExtractor:
    def __init__(self, debug = False):
        self.debug = debug

    def sliding_window(self, signal, rate, window_length_in_ms, skip_in_ms):
        """
        Return slices of a signal based on a sliding winding concept with a 
        defined window width and window step.
        
        Args: 	    
            sig       (array) : audio signal (list of float)
            rate        (int) : sampling rate (= average number of samples pro 1 second)
            window_length_in_ms (int) : slicing window in milli-seconds.
            skip_in_ms          (int) : slicing step in milli-seconds.

        Returns: 	    
            (array) : array window/slice.
        """
        window_length = int(rate / 1000) * window_length_in_ms 
        window_skip   = int(rate / 1000) * skip_in_ms
        
        if window_skip is None: window_skip = window_length
        for i in range(0, len(signal) - int(window_length), int(window_skip)):
                yield signal[i: i + window_length]
                
    def split_signal_in_frames(self, signal, rate, frame_size_in_ms = 100):
        """
        Return slices of a signal based on a sliding winding concept with a 
        fixed window width and window step.
        
        Args: 	    
            sig       (array) : audio signal (list of float)
            rate        (int) : sampling rate (= average number of samples pro 1 second)
            frame_size_in_ms (int) : slicing window and step in milli-seconds.

        Returns: 	    
            (list) : list of signal slices.
        """
        samples_pro_frame       = int(rate / 1000) * frame_size_in_ms
        samples_count_pro_frame = int(int(len(signal) / samples_pro_frame) * samples_pro_frame)
        frames                  = numpy.split(signal[:samples_count_pro_frame],
                                           len(signal[:samples_count_pro_frame]) / int(samples_count_pro_frame))
        try   : frames.append(signal[samples_count_pro_frame:])
        except: pass
        return frames  
    
    def get_dominant_frequencies(self, slices, rate):
        """
        Returns the dominant audio frequency (in Hertz) of a given audio signal.
            
        Args: 	    
            slices  (list) : list of signal slices.
            rate     (int) : sampling rate (= average number of samples pro 1 second)

        Returns: 	    
            (array) : array of dominant frequencies.
        """        
        dominant_frequencies = []
        for sig in [sig for sig in slices if sig.size > 0]:
            fourrier_transform = numpy.fft.fft(sig)
            psd                = (1 / len(fourrier_transform)) * abs(fourrier_transform)**2 
            frequencies        = numpy.fft.fftfreq(sig.size , 1 / rate)
            frequencies        = numpy.array([freq for freq in frequencies if freq >= 0])
            idx                = numpy.argsort(frequencies)
            dominant_frequencies.append(frequencies[idx][numpy.argmax(psd[idx])])
            
            # debug: print and plot results
            if self.debug:
                print("Dominant Frequency: %5f KHz" % round(dominant_frequencies[-1]/1000, 3))
                plt.plot(frequencies[idx] / 1000, psd[idx])
                plt.plot(frequencies[numpy.argmax(psd[idx])] /1000, psd[idx].max(), "rx")
                plt.ylabel('Power [dB]')
                plt.xlabel('Frequencies [KHz]')
                plt.ylim(psd.min(), psd.max() + .1 * psd.max())
                plt.xlim(-.25, dominant_frequencies[-1] /1000 + .25*dominant_frequencies[-1] /1000)
                plt.show()
        return numpy.array(dominant_frequencies)
        
    def main(self, signal, rate):
        """
        Returns the dominant audio frequencies of a given wave file.
            
        Args: 	    
            file_name (str) : name of an audio file name.

        Returns: 	    
            (array) : array of dominant frequencies.
        """     
        
        # apply Band pass Butterworth filter
        lower_cutoff, upper_cutoff = 50, 3000
        b, a   = scipy.signal.butter(6, [(lower_cutoff * 2) / rate , (upper_cutoff * 2) / rate ], 'band')
        w, h   = scipy.signal.freqs(b, a, len(signal))
        signal = scipy.signal.lfilter(b, a, signal) 

        # compute dominant frequencies               
        slices    = [w for w in self.sliding_window(signal, rate, 10, 5)] 
        dom_freqs = self.get_dominant_frequencies(slices, rate)
        return dom_freqs
    

if __name__ == '__main__':
    # read audio data
    signal, rate = librosa.load("test.wav")

    # test dominant frequencies extraction
    dominant_frequencies_extractor = DominantFrequenciesExtractor(debug=False)
    dominant_frequencies           = dominant_frequencies_extractor.main(signal, rate)
    
    #  test fundamental frequencies extraction
    fundamental_frequencies = FundamentalFrequenciesExtractor(False)
    pitches, harmonic_rates, argmins, times = fundamental_frequencies.main(rate, signal)
