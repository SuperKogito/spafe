import time
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


class FundamentalFrequencies:
    def __init__(self):
        pass

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
#        x = np.array(x, np.float64)
#        w = x.size
#        tau_max  = min(tau_max, w)
#        x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
#        size = w + tau_max
#        p2   = (size // 32).bit_length()
#        nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
#        size_pad     = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
#        fc   = np.fft.rfft(x, size_pad)
#        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
#        return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv
        x = np.array(x, np.float64)
        w = x.size
        x_cumsum = np.concatenate((np.array([0]), (x * x).cumsum()))
        conv = fftconvolve(x, x[::-1])
        tmp = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
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
    
        cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float) # scipy method
        return np.insert(cmndf, 0, 1)
    
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
        
        Parameters
        ----------
        sig: Audio signal (list of float)
        sr: sampling rate (int)
        w_len: size of the analysis window (samples)
        w_step: size of the lag between two consecutives windows (samples)
        f0_min: Minimum fundamental frequency that can be detected (hertz)
        f0_max: Maximum fundamental frequency that can be detected (hertz)
        harmo_tresh: Threshold of detection. The yalgorithmù return the first minimum of the CMND fubction below this treshold.
        :returns:
            * pitches: list of fundamental frequencies,
            * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
            * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
            * times: list of time of each estimation
        :rtype: tuple
        """
        print('Yin: compute yin algorithm')
        tau_min = int(sr / f0_max)
        tau_max = int(sr / f0_min)
    
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
            if np.argmin(cmdf) > tau_min:
                argmins[i] = float(sr / np.argmin(cmdf))
            #  A pitch was found
            if p != 0: 
                pitches[i]        = float(sr / p)
                harmonic_rates[i] = cmdf[p]
            #  No pitch, but we compute a value of the harmonic rate
            else: 
                harmonic_rates[i] = min(cmdf)
                
        return pitches, harmonic_rates, argmins, times
    
    def main(self, sr, sig, w_len=1024, w_step=256, f0_min=70, f0_max=200, harmo_thresh=0.85, audioDir="./", dataFileName=None, verbose=4):
        """
        Run the computation of the Yin algorithm on a example file.
        Write the results (pitches, harmonic rates, parameters ) in a numpy file.
        audioFileName: name of the audio file
        :type audioFileName: str
        w_len: length of the window
        :type wLen: int
        wStep: length of the "hop" size
        :type wStep: int
        f0_min: minimum f0 in Hertz
        :type f0_min: float
        f0_max: maximum f0 in Hertz
        :type f0_max: float
        harmo_thresh: harmonic threshold
        :type harmo_thresh: float
        audioDir: path of the directory containing the audio file
        :type audioDir: str
        dataFileName: file name to output results
        :type dataFileName: str
        verbose: Outputs on the console : 0-> nothing, 1-> warning, 2 -> info, 3-> debug(all info), 4 -> plot + all info
        :type verbose: int
        """
        start = time.time()
        pitches, harmonic_rates, argmins, times = self.compute_yin(sig, sr, dataFileName, w_len, w_step, f0_min, f0_max, harmo_thresh)
        print ("Yin computed in: ", time.time() - start)
    
        duration = len(sig)/float(sr)
        plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left   = 0.125,  right  = 0.9,   
                            bottom = 0.1,    top    = 0.9,     
                            wspace = 0.2,    hspace = 0.99)
        
        if verbose >3:
            ax1 = plt.subplot(4, 1, 1)
            ax1.plot([float(x) * duration / len(sig) for x in range(0, len(sig))], sig)
            ax1.set_title('Audio data')
            ax1.set_ylabel('Amplitude')
            ax2 = plt.subplot(4, 1, 2)
            ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
            ax2.set_title('F0')
            ax2.set_ylabel('Frequency (Hz)')
            ax3 = plt.subplot(4, 1, 3, sharex=ax2)
            ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates, "-x")
            ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harmo_thresh] * len(harmonic_rates), 'r', "--")
            ax3.set_title('Harmonic rate')
            ax3.set_ylabel('Rate')
            ax4 = plt.subplot(4, 1, 4, sharex=ax2)
            ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins, "-x")
            ax4.set_title('Index of minimums of CMND')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.set_xlabel('Time (seconds)')
            plt.show()
    
        return np.array(pitches), harmonic_rates, argmins, times, duration

if __name__ == '__main__':
    #  read signal, convert from bits and select from signal 
    freq, sig               = wavfile.read('sample2.wav')
    sig = sig[:100000]
    fundamental_frequencies = FundamentalFrequencies()
    pitches, harmonic_rates, argmins, times, duration = fundamental_frequencies.main(freq, sig)

    #  prepare time array
    time_array  = np.arange(0, sig.shape[0], 1) / freq
    time_array *= 1000                                   # scale to milliseconds
    
    #  plot the tone
    plt.plot(time_array[::10], sig[::10])
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.show()
    
    #  comput fft and psd
    N                  = len(sig) 
    fourrier_transform = np.fft.fft(sig) #  take the fourier transform 
    psd                = (1 / N)* abs(fourrier_transform)**2  #  square it to get the power 
    freq_array         = np.arange(0, N, 1.0) * (freq / N)
   
    #  plot power density
    plt.plot(freq_array[::100], 10*np.log10(fourrier_transform)[::100], "-")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.show()

    #  plot spectogram    
    plt.specgram(sig, Fs=freq)
    plt.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0,5000)
    plt.show()
    
    plt.figure(figsize=(20, 10))
    duration = len(sig)/float(freq)
    plt.stem([float(x) * duration / len(pitches) for x in range(0, len(pitches))][::], pitches[::],  markerfmt=' ',use_line_collection=True)
    plt.title('F0')
    plt.ylabel('Frequency (Hz)')
    plt.show()
