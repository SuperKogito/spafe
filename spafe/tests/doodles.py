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
    


def gammatonegram(x, sr=20000, twin=0.025, thop=0.010, N=64,
                  fmin=50, fmax=10000, width=1.0):
    """
    # Ellis' description in MATLAB:
    # [Y,F] = gammatonegram(X,SR,N,TWIN,THOP,FMIN,FMAX,USEFFT,WIDTH)
    # Calculate a spectrogram-like time frequency magnitude array
    # based on Gammatone subband filters.  Waveform X (at sample
    # rate SR) is passed through an N (default 64) channel gammatone
    # auditory model filterbank, with lowest frequency FMIN (50)
    # and highest frequency FMAX (SR/2).  The outputs of each band
    # then have their energy integrated over windows of TWIN secs
    # (0.025), advancing by THOP secs (0.010) for successive
    # columns.  These magnitudes are returned as an N-row
    # nonnegative real matrix, Y.
    # WIDTH (default 1.0) is how to scale bandwidth of filters
    # relative to ERB default (for fast method only).
    # F returns the center frequencies in Hz of each row of Y
    # (uniformly spaced on a Bark scale).
    # 2009/02/23 DAn Ellis dpwe@ee.columbia.edu
    # Sat May 27 15:37:50 2017 Maddie Cusimano mcusi@mit.edu, converted to python
    """

    # Entirely skipping Malcolm's function, because would require
    # altering ERBFilterBank code as well.
    # i.e., in Ellis' code: usefft = 1
    #assert (x.dtype == 'int16')

    # How long a window to use relative to the integration window requested
    winext = 1;
    twinmod = winext * twin;
    nfft = int(2 ** (np.ceil(np.log(2 * twinmod * sr) / np.log(2))))
    nhop = int(np.round(thop * sr))
    nwin = int(np.round(twinmod * sr))
    gtm = gammatone_filter_banks(N, nfft, sr, fmin, fmax)
    # perform FFT and weighting in amplitude domain
    # note: in MATLAB, abs(spectrogram(X, hanning(nwin), nwin-nhop, nfft, SR))
    #                  = abs(specgram(X,nfft,SR,nwin,nwin-nhop))
    # in python approx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin,
    #                    noverlap=nwin-nhop, nfft=nfft, detrend=False,
    #                    scaling='density', mode='magnitude')
    plotF, plotT, Sxx = sps.spectrogram(x, 
                                        fs       = sr, 
                                        window   = 'hann', 
                                        nperseg  = nwin,
                                        noverlap = nwin - nhop, 
                                        nfft     = nfft, 
                                        detrend  = False,
                                        scaling  = 'density',
                                        mode     = 'magnitude')
    y = (1 / nfft) * np.dot(gtm, Sxx)
    return y

def get_gfcc(audio, rate, log_constant=1e-80, db_threshold=-50.):
    sxx = gammatonegram(audio, sr=rate, fmin=20, fmax=int(rate / 2.))
    sxx[sxx == 0] = log_constant
    sxx = 20.0 * np.log10(sxx)  # to db
    sxx[sxx < db_threshold] = db_threshold
    # center_freq
    return sxx
