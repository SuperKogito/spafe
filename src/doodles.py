def split_signal_in_10ms_frames(self, signal, rate):
    samples_pro_10ms_frame  = int(rate / 100)
    samples_count_pro_frame = int(int(len(signal) / samples_pro_10ms_frame)*samples_pro_10ms_frame)
    frames                  = np.split(signal[:samples_count_pro_frame],
                                       len(signal[:samples_count_pro_frame]) / int(samples_pro_10ms_frame))
    try   : frames.append(signal[samples_count_pro_frame:])
    except: pass
    return frames  

def get_dominant_frequencies(self, slices, rate):
    """
    Returns the dominant audio frequency of a given frame in hertz.
    """
    dominant_frequencies = []
    for sig in [sig for sig in slices if sig != []]:
        fourrier_transform = np.fft.fft(sig)
        psd                = (1 / len(fourrier_transform)) * abs(fourrier_transform)**2 
        frequencies        = np.fft.fftfreq(sig.size, 1 / rate)
        idx                = np.argsort(frequencies)
        dominant_frequencies.append(frequencies[np.argmax(psd)])
        # plot
        plt.stem(freqs[idx][::10], psd[idx][::10])
        plt.plot(0, frequencies[np.argmax(psd)], "ro")
        plt.ylabel('Power [dB]')
        plt.xlabel('Frequencies [Hz]')
        plt.ylim(np.min(psd), np.max(psd))
        plt.show()  
                              
    return np.array(dominant_frequencies)
  

def main(self, file_name):
    signal, rate = librosa.load(file_name)
    
