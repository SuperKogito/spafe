import scipy
import librosa
import numpy as np
import matplotlib.pyplot as plt

class DominantFrequenciesExtractor:
    def __init__(self, debug = False):
        self.debug = debug

    def sliding_window(self, signal, rate, window_length_in_ms, skip_in_ms):
        window_length = int(rate / 1000) * window_length_in_ms  
        window_skip   = int(rate / 1000) * skip_in_ms
        
        if window_skip is None: window_skip = window_length
        for i in range(0, len(signal) - int(window_length), int(window_skip)):
                yield signal[i: i + window_length]
                
    def split_signal_in_frames(self, signal, rate, frame_size_in_ms = 100):
        samples_pro_frame       = int(rate / 1000) * frame_size_in_ms
        samples_count_pro_frame = int(int(len(signal) / samples_pro_frame) * samples_pro_frame)
        frames                  = np.split(signal[:samples_count_pro_frame],
                                           len(signal[:samples_count_pro_frame]) / int(samples_count_pro_frame))
        try   : frames.append(signal[samples_count_pro_frame:])
        except: pass
        return frames  
    
    def get_dominant_frequencies(self, slices, rate):
        """
        Returns the dominant audio frequency of a given frame in hertz.
        """        
        dominant_frequencies = []
        for sig in [sig for sig in slices if sig.size > 0]:
            fourrier_transform = np.fft.fft(sig)
            psd                = (1 / len(fourrier_transform)) * abs(fourrier_transform)**2 
            frequencies        = np.fft.fftfreq(sig.size , 1 / rate)
            frequencies        = np.array([freq for freq in frequencies if freq >= 0])
            idx                = np.argsort(frequencies)
            dominant_frequencies.append(frequencies[idx][np.argmax(psd[idx])])
            
            # debug: print and plot results
            if self.debug:
                print("Dominant Frequency: %5f KHz" % round(dominant_frequencies[-1]/1000, 3))
                plt.plot(frequencies[idx] / 1000, psd[idx])
                plt.plot(frequencies[np.argmax(psd[idx])] /1000, psd[idx].max(), "rx")
                plt.ylabel('Power [dB]')
                plt.xlabel('Frequencies [KHz]')
                plt.ylim(psd.min(), psd.max() + .1 * psd.max())
                plt.xlim(-.25, dominant_frequencies[-1] /1000 + .25*dominant_frequencies[-1] /1000)
                plt.show()
                
        return np.array(dominant_frequencies)
        
    def main(self, file_name):
        signal, rate = librosa.load(file_name)
        signal       = signal[:rate*20:100]
        
        # apply Band pass Butterworth filter
        lower_cutoff, upper_cutoff = 50, 3000
        b, a   = scipy.signal.butter(6, [(lower_cutoff * 2) / rate , (upper_cutoff * 2) / rate ], 'band')
        w, h   = scipy.signal.freqs(b, a, len(signal))
        signal = scipy.signal.lfilter(b, a, signal) 

        # compute dominant frequencies               
        slices    = self.split_signal_in_frames(signal, rate)
        slices    = [w for w in self.sliding_window(signal, rate, 10, 5)] 
        dom_freqs = self.get_dominant_frequencies(slices, rate)
        return dom_freqs
    
# test 
dominant_frequencies_extractor = DominantFrequenciesExtractor(debug=True)
dominant_frequencies           = dominant_frequencies_extractor.main("voc0U0t10dfnzl.wav")